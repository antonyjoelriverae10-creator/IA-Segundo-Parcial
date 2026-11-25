// main.cpp - punto de entrada del controlador "locomotion".
// Solo orquesta: inicializa, sondea la fuente de ordenes y avanza la locomocion
// un timestep por iteracion. No contiene logica de marcha.
#include <webots/Robot.hpp>

#include <cstdio>
#include <cstdlib>
#include <exception>

#include "CustomDataSource.hpp"
#include "Locomotion.hpp"

namespace {

// Vacia las ordenes pendientes sin bloquear (tope por ciclo por seguridad).
void pollCommands(loco::CommandSource &source, loco::Locomotion &locomotion) {
  loco::Command command;
  for (int i = 0; i < 16 && source.poll(command); ++i) locomotion.submit(command);
}

}  // namespace

int main() {
  try {
    // Orden de declaracion importante: Robot primero => se destruye el ultimo.
    webots::Robot robot;

    loco::LocomotionConfig config;
    // Si el Supervisor deja de enviar ordenes durante 3 s, el robot se detiene.
    // El Supervisor envia una orden nueva en cada decision (~0.5 s).
    config.commandTimeoutMs = 3000;

    loco::Locomotion locomotion(config);
    if (!locomotion.initialize(robot)) {
      std::fprintf(stderr, "[locomotion] initialization failed, controller exits\n");
      return EXIT_FAILURE;
    }

    loco::CustomDataSource source(robot);
    do {
      pollCommands(source, locomotion);
    } while (locomotion.step());
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[locomotion] fatal exception: %s\n", e.what());
    return EXIT_FAILURE;
  } catch (...) {
    std::fprintf(stderr, "[locomotion] fatal unknown exception\n");
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
