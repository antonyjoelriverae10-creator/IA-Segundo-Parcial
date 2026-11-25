// Locomotion.hpp - capa de locomocion independiente para el DARwIn-OP en Webots.
//
// Separacion de responsabilidades:
//   * CommandSource : RECEPCION de ordenes (hoy no hay ninguna; manana un Receiver
//                     del Supervisor, una cola, etc.). No conoce al robot.
//   * Locomotion    : EJECUCION. Valida dispositivos, posee los dos gestores
//                     oficiales y avanza una maquina de estados un timestep por
//                     llamada a step(). No sabe de donde vienen las ordenes.
//
// Todo corre en un unico hilo (el del controlador de Webots): no hay mutex porque
// no hace falta; un transporte futuro debe encolar ordenes desde ese mismo hilo.
#ifndef LOCOMOTION_HPP
#define LOCOMOTION_HPP

#include <memory>
#include <string>

namespace webots {
class Robot;
}

// Declaraciones adelantadas (igual que walk.hpp oficial): las cabeceras pesadas
// solo se incluyen en Locomotion.cpp. Verificar que el espacio de nombres
// "managers" coincide con RobotisOp2*Manager.hpp (ver notas de incertidumbre).
namespace managers {
class RobotisOp2GaitManager;
class RobotisOp2MotionManager;
}

namespace loco {

// Limita a [-1, 1]; NaN se trata como 0 (una orden corrupta nunca mueve el robot).
double clampUnit(double value);

// Orden de alto nivel. "forward" > 0 avanza, < 0 retrocede; "turn" > 0 gira a la
// izquierda, < 0 a la derecha (convencion del ejemplo walk; verificar en simulacion).
struct Command {
  enum class Type { Stop, Move };

  Command() : type(Type::Stop), forward(0.0), turn(0.0) {}

  static Command stop();
  static Command move(double forward, double turn);
  static Command advance(double amplitude);    // avance (usa |amplitude|)
  static Command retreat(double amplitude);    // retroceso (usa |amplitude|)
  static Command turnLeft(double amplitude);   // giro en el sitio a la izquierda
  static Command turnRight(double amplitude);  // giro en el sitio a la derecha

  Type type;
  double forward;
  double turn;
};

// Interfaz de recepcion. poll() NO debe bloquear: devuelve true y rellena "out"
// si hay una orden pendiente, false si no hay ninguna.
class CommandSource {
 public:
  virtual ~CommandSource() {}
  virtual bool poll(Command &out) = 0;
};

struct LocomotionConfig {
  std::string gaitConfigFile = "config.ini";  // relativo al directorio del controlador
  int initialPage = 9;           // pagina de postura inicial (la usa el ejemplo walk)
  bool balanceEnable = true;
  bool moveAimOn = false;
  int poseTimeoutMs = 5000;      // maximo para completar la postura inicial
  int startSettleMs = 200;       // tiempo de asentamiento tras gait->start()
  int stopSettleMs = 200;        // tiempo de asentamiento tras gait->stop()
  int commandTimeoutMs = 0;      // >0: detener si no llegan ordenes en ese tiempo
  double zeroEpsilon = 1e-3;     // amplitudes menores se consideran "parado"
};

class Locomotion {
 public:
  enum class State {
    Uninitialized,  // antes de initialize()
    Booting,        // primer timestep para que los sensores tengan datos
    Posing,         // ejecutando la postura inicial (no camina)
    Idle,           // de pie, sin caminar
    Starting,       // gait->start() llamado, asentandose
    Walking,        // marcha activa
    Stopping,       // gait->stop() llamado, asentandose
    Faulted         // error: no se comanda nada mas
  };

  explicit Locomotion(const LocomotionConfig &config = LocomotionConfig());
  ~Locomotion();

  Locomotion(const Locomotion &) = delete;
  Locomotion &operator=(const Locomotion &) = delete;

  // Valida dispositivos y gestores. NO mueve el robot ni avanza la simulacion.
  // "robot" debe sobrevivir a este objeto (declararlo antes en main).
  bool initialize(webots::Robot &robot);

  // Registra la orden mas reciente (la ultima gana). No toca los gestores: solo
  // se aplica en el siguiente step().
  void submit(const Command &command);

  // Un ciclo: ejecuta locomocion y luego avanza UN timestep de Webots.
  // Devuelve false cuando Webots termina la simulacion (o si no esta inicializado).
  bool step();

  State state() const { return state_; }
  static const char *stateName(State state);

 private:
  bool acquireDevices(webots::Robot &robot);
  bool fail(const char *message);
  void enterState(State next);
  void execute();
  bool isStopRequest() const;

  LocomotionConfig config_;
  webots::Robot *robot_;
  std::unique_ptr<managers::RobotisOp2MotionManager> motion_;
  std::unique_ptr<managers::RobotisOp2GaitManager> gait_;

  State state_;
  int timeStepMs_;
  int stateElapsedMs_;
  int settleRemainingMs_;
  int sinceLastCommandMs_;
  bool poseStarted_;

  double desiredForward_;
  double desiredTurn_;
};

}  // namespace loco

#endif  // LOCOMOTION_HPP
