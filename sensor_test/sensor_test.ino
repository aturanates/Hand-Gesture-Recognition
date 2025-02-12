void setup() {

  Serial.begin(115200);
}


void loop() {
  int sensorValue0 = analogRead(A0);
  int sensorValue1 = analogRead(A1);
  int sensorValue2 = analogRead(A2);
  int sensorValue3 = analogRead(A3);
 
  Serial.print(sensorValue0);
  Serial.print(",");
  Serial.print(sensorValue1);
  Serial.print(",");
  Serial.print(sensorValue2);
  Serial.print(",");
  Serial.println(sensorValue3);
  
  /*
  Serial.print(sensorValue0);
  Serial.print(",");
  Serial.print(0);
  Serial.print(",");
  Serial.print(0);
  Serial.print(",");
  Serial.println(0);
  */
  delay(15);
}

