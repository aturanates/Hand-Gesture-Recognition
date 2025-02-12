// Example for MCP3008 using the MCP3008 library
#include <Adafruit_MCP3008.h>

Adafruit_MCP3008 adc;

void setup() {
  Serial.begin(9600);
}

void loop() {

  adc.begin(10);
  
  int value = adc.readADC(0);
  int value2 = adc.readADC(1);
  int value3 = adc.readADC(2);
  int value4 = adc.readADC(3);
  
  Serial.print(value);
  Serial.print(",");
  Serial.print(value2);
  Serial.print(",");
  Serial.print(value3);
  Serial.print(",");
  Serial.println(value4);
  
  delay(10);  // Adjust delay as needed
}
