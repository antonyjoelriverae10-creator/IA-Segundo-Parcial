// CustomDataSource.hpp - recibe ordenes del Supervisor a traves del campo
// "customData" del propio nodo Robot (sin Emitter/Receiver ni dispositivos extra).
// Formato escrito por el Supervisor:  "<seq>,<forward>,<turn>"   p. ej. "17,1.0,0.0"
// Una orden solo se entrega cuando cambia <seq> (asi el vigilante de Locomotion
// distingue un latido nuevo de un valor viejo). El seq que ya hubiera al arrancar
// se ignora: tras restartController() no se reaplican ordenes obsoletas.
#ifndef CUSTOM_DATA_SOURCE_HPP
#define CUSTOM_DATA_SOURCE_HPP

#include <webots/Robot.hpp>

#include <cstdio>
#include <string>

#include "Locomotion.hpp"

namespace loco {

class CustomDataSource : public CommandSource {
 public:
  explicit CustomDataSource(webots::Robot &robot) : robot_(robot), lastSeq_(-1) {
    long seq;
    double forward, turn;
    if (parse(robot_.getCustomData(), seq, forward, turn)) lastSeq_ = seq;
  }

  bool poll(Command &out) override {
    long seq;
    double forward, turn;
    if (!parse(robot_.getCustomData(), seq, forward, turn)) return false;
    if (seq == lastSeq_) return false;
    lastSeq_ = seq;
    out = Command::move(forward, turn);  // (0,0) equivale a parar en Locomotion
    return true;
  }

 private:
  static bool parse(const std::string &text, long &seq, double &forward, double &turn) {
    return std::sscanf(text.c_str(), "%ld,%lf,%lf", &seq, &forward, &turn) == 3;
  }

  webots::Robot &robot_;
  long lastSeq_;
};

}  // namespace loco

#endif  // CUSTOM_DATA_SOURCE_HPP
