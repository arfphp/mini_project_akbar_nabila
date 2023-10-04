#include <Firmata.h>

int LED1_PIN = D5;
int LED2_PIN = D3;

void setup()
{
  Firmata.setFirmwareVersion(0, 1);
  Firmata.attach(ANALOG_MESSAGE, analogWriteCallback);
  Firmata.attach(DIGITAL_MESSAGE, digitalWriteCallback);
  Firmata.begin(57600);

  pinMode(LED1_PIN, OUTPUT);
  pinMode(LED2_PIN, OUTPUT);
}

void loop()
{
  while (Firmata.available())
  {
    Firmata.processInput();
  }
}

void analogWriteCallback(byte pin, int value)
{
  if (pin == LED1_PIN)
  {
    analogWrite(LED1_PIN, value);
  }
  else if (pin == LED2_PIN)
  {
    analogWrite(LED2_PIN, value);
  }
}

void digitalWriteCallback(byte pin, int value)
{
  if (pin == LED1_PIN)
  {
    digitalWrite(LED1_PIN, value);
  }
  else if (pin == LED2_PIN)
  {
    digitalWrite(LED2_PIN, value);
  }
}