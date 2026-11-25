// Locomotion.cpp - implementacion. Ver Locomotion.hpp para el diseno general.
#include "Locomotion.hpp"

#include <webots/Accelerometer.hpp>
#include <webots/Gyro.hpp>
#include <webots/Motor.hpp>
#include <webots/PositionSensor.hpp>
#include <webots/Robot.hpp>

#include <RobotisOp2GaitManager.hpp>
#include <RobotisOp2MotionManager.hpp>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <new>

namespace loco {

namespace {

// Los 20 motores del DARwIn-OP; el sensor de posicion de cada uno se llama igual
// con una "S" al final (convencion del modelo oficial).
const char *const kMotorNames[] = {
    "ShoulderR", "ShoulderL", "ArmUpperR", "ArmUpperL", "ArmLowerR",
    "ArmLowerL", "PelvYR",    "PelvYL",    "PelvR",     "PelvL",
    "LegUpperR", "LegUpperL", "LegLowerR", "LegLowerL", "AnkleR",
    "AnkleL",    "FootR",     "FootL",     "Neck",      "Head"};
const int kMotorCount = static_cast<int>(sizeof(kMotorNames) / sizeof(kMotorNames[0]));

// El gestor de marcha trabaja en sub-ciclos de 8 ms: no admite timesteps menores.
const int kMinTimeStepMs = 8;

void logInfo(const char *fmt, const char *a, const char *b) {
  std::printf("[locomotion] ");
  std::printf(fmt, a, b);
  std::printf("\n");
  std::fflush(stdout);
}

void logError(const char *message) {
  std::fprintf(stderr, "[locomotion] ERROR: %s\n", message);
  std::fflush(stderr);
}

}  // namespace

// ---------------------------------------------------------------- Command ----

double clampUnit(double value) {
  if (std::isnan(value)) return 0.0;
  if (value > 1.0) return 1.0;
  if (value < -1.0) return -1.0;
  return value;
}

Command Command::stop() { return Command(); }

Command Command::move(double forward, double turn) {
  Command c;
  c.type = Type::Move;
  c.forward = clampUnit(forward);
  c.turn = clampUnit(turn);
  return c;
}

Command Command::advance(double amplitude) { return move(std::fabs(amplitude), 0.0); }
Command Command::retreat(double amplitude) { return move(-std::fabs(amplitude), 0.0); }
Command Command::turnLeft(double amplitude) { return move(0.0, std::fabs(amplitude)); }
Command Command::turnRight(double amplitude) { return move(0.0, -std::fabs(amplitude)); }

// ------------------------------------------------------------- Locomotion ----

Locomotion::Locomotion(const LocomotionConfig &config)
    : config_(config),
      robot_(nullptr),
      state_(State::Uninitialized),
      timeStepMs_(0),
      stateElapsedMs_(0),
      settleRemainingMs_(0),
      sinceLastCommandMs_(0),
      poseStarted_(false),
      desiredForward_(0.0),
      desiredTurn_(0.0) {}

// Definido aqui (no en el .hpp) porque unique_ptr necesita el tipo completo.
// Los gestores se destruyen antes que el Robot (que main declara primero).
// No se llama a gait->stop(): si step() devolvio false la simulacion ya termina
// y no queda nada que comandar.
Locomotion::~Locomotion() {
  gait_.reset();
  motion_.reset();
}

const char *Locomotion::stateName(State state) {
  switch (state) {
    case State::Uninitialized: return "Uninitialized";
    case State::Booting: return "Booting";
    case State::Posing: return "Posing";
    case State::Idle: return "Idle";
    case State::Starting: return "Starting";
    case State::Walking: return "Walking";
    case State::Stopping: return "Stopping";
    case State::Faulted: return "Faulted";
  }
  return "Unknown";
}

bool Locomotion::fail(const char *message) {
  logError(message);
  gait_.reset();
  motion_.reset();
  state_ = State::Faulted;
  return false;
}

// Comprueba TODOS los dispositivos antes de usar ninguno y los habilita.
// Los gestores oficiales leen giroscopio y sensores de posicion; habilitarlos
// aqui es idempotente si los gestores ya lo hacen.
bool Locomotion::acquireDevices(webots::Robot &robot) {
  webots::Accelerometer *accelerometer = robot.getAccelerometer("Accelerometer");
  webots::Gyro *gyro = robot.getGyro("Gyro");
  if (accelerometer == nullptr) return fail("device 'Accelerometer' not found");
  if (gyro == nullptr) return fail("device 'Gyro' not found");

  for (int i = 0; i < kMotorCount; ++i) {
    const std::string motorName = kMotorNames[i];
    const std::string sensorName = motorName + "S";
    if (robot.getMotor(motorName) == nullptr) {
      logInfo("device motor '%s' not found%s", motorName.c_str(), "");
      return fail("missing motor (see message above)");
    }
    webots::PositionSensor *sensor = robot.getPositionSensor(sensorName);
    if (sensor == nullptr) {
      logInfo("device position sensor '%s' not found%s", sensorName.c_str(), "");
      return fail("missing position sensor (see message above)");
    }
    sensor->enable(timeStepMs_);
  }
  accelerometer->enable(timeStepMs_);
  gyro->enable(timeStepMs_);
  return true;
}

bool Locomotion::initialize(webots::Robot &robot) {
  if (state_ != State::Uninitialized) {
    logError("initialize() called twice");
    return false;
  }
  robot_ = &robot;

  timeStepMs_ = static_cast<int>(std::lround(robot.getBasicTimeStep()));
  if (timeStepMs_ < kMinTimeStepMs) return fail("basicTimeStep < 8 ms is not supported by the gait manager");

  if (!acquireDevices(robot)) return false;  // fail() ya registro el motivo

  // Comprobacion previa del archivo de marcha para dar un error claro; el gestor
  // oficial lo abre por su cuenta (ruta relativa al directorio del controlador).
  {
    std::ifstream iniFile(config_.gaitConfigFile.c_str());
    if (!iniFile.good()) return fail("gait config file not found (expected next to the controller)");
  }

  motion_.reset(new (std::nothrow) managers::RobotisOp2MotionManager(&robot));
  if (!motion_) return fail("could not allocate RobotisOp2MotionManager");
  if (!motion_->isCorrectlyInitialized()) return fail("RobotisOp2MotionManager not correctly initialized");

  gait_.reset(new (std::nothrow) managers::RobotisOp2GaitManager(&robot, config_.gaitConfigFile));
  if (!gait_) return fail("could not allocate RobotisOp2GaitManager");
  if (!gait_->isCorrectlyInitialized()) return fail("RobotisOp2GaitManager not correctly initialized");

  // Configuracion de la marcha (aun sin llamar a start(): el robot no camina).
  gait_->setXAmplitude(0.0);
  gait_->setYAmplitude(0.0);
  gait_->setAAmplitude(0.0);
  gait_->setMoveAimOn(config_.moveAimOn);
  gait_->setBalanceEnable(config_.balanceEnable);

  enterState(State::Booting);
  return true;
}

void Locomotion::submit(const Command &command) {
  if (command.type == Command::Type::Stop) {
    desiredForward_ = 0.0;
    desiredTurn_ = 0.0;
  } else {
    desiredForward_ = clampUnit(command.forward);
    desiredTurn_ = clampUnit(command.turn);
  }
  sinceLastCommandMs_ = 0;
}

bool Locomotion::isStopRequest() const {
  return std::fabs(desiredForward_) < config_.zeroEpsilon &&
         std::fabs(desiredTurn_) < config_.zeroEpsilon;
}

void Locomotion::enterState(State next) {
  if (next != state_) logInfo("%s -> %s", stateName(state_), stateName(next));
  state_ = next;
  stateElapsedMs_ = 0;
}

void Locomotion::execute() {
  // Vigilante: si el agente desaparece, el robot no sigue caminando indefinidamente.
  if (sinceLastCommandMs_ < 1000000000) sinceLastCommandMs_ += timeStepMs_;
  if (config_.commandTimeoutMs > 0 && sinceLastCommandMs_ > config_.commandTimeoutMs) {
    desiredForward_ = 0.0;
    desiredTurn_ = 0.0;
  }
  stateElapsedMs_ += timeStepMs_;

  switch (state_) {
    case State::Uninitialized:
    case State::Faulted:
      break;

    case State::Booting:
      // El primer robot->step() (al final de step()) aun no ha ocurrido: se deja
      // pasar un ciclo para que los sensores tengan valores antes de la postura.
      enterState(State::Posing);
      poseStarted_ = false;
      break;

    case State::Posing:
      // La postura solo se pide aqui, tras haber comprobado el gestor de movimientos
      // en initialize(). sync=false: no bloquea; se avanza paso a paso en el bucle.
      if (!poseStarted_) {
        motion_->playPage(config_.initialPage, false);
        poseStarted_ = true;
        stateElapsedMs_ = 0;
      }
      motion_->step(timeStepMs_);
      if (!motion_->isMotionPlaying()) {
        enterState(State::Idle);
      } else if (stateElapsedMs_ >= config_.poseTimeoutMs) {
        fail("initial pose did not finish before timeout");
      }
      break;

    case State::Idle:
      // Las ordenes recibidas durante Booting/Posing quedan guardadas y se
      // aplican aqui. El gestor de movimientos ya no se usa mientras camina.
      if (!isStopRequest()) {
        gait_->setXAmplitude(0.0);
        gait_->setAAmplitude(0.0);
        gait_->start();
        settleRemainingMs_ = config_.startSettleMs;
        enterState(State::Starting);
      }
      break;

    case State::Starting:
      gait_->step(timeStepMs_);  // amplitudes a 0: arranque sin sacudidas
      settleRemainingMs_ -= timeStepMs_;
      if (settleRemainingMs_ <= 0) enterState(State::Walking);
      break;

    case State::Walking:
      if (isStopRequest()) {
        gait_->setXAmplitude(0.0);
        gait_->setAAmplitude(0.0);
        gait_->stop();
        settleRemainingMs_ = config_.stopSettleMs;
        enterState(State::Stopping);
      } else {
        gait_->setXAmplitude(desiredForward_);  // ya limitadas a [-1, 1]
        gait_->setAAmplitude(desiredTurn_);
      }
      gait_->step(timeStepMs_);
      break;

    case State::Stopping:
      gait_->step(timeStepMs_);  // deja que la marcha termine su ciclo
      settleRemainingMs_ -= timeStepMs_;
      if (settleRemainingMs_ <= 0) enterState(State::Idle);
      break;
  }
}

bool Locomotion::step() {
  if (robot_ == nullptr || state_ == State::Uninitialized) return false;
  execute();
  // Unico punto que avanza el reloj de Webots; -1 indica fin de simulacion.
  return robot_->step(timeStepMs_) != -1;
}

}  // namespace loco
