// good for 24000 dps
uint16_t accelbuffZ[8];
void setup()
{
  SerialUSB.begin(0);
  while(!SerialUSB);
  analogReadResolution(12);
//  Serial.begin(115200);
  delay(500);

}

void loop()
{
    while (true) {
      accelbuffZ[0]=analogRead(0);
      accelbuffZ[1]=analogRead(1);
      accelbuffZ[2]=analogRead(2);
      accelbuffZ[3]=analogRead(3);
      accelbuffZ[4]=analogRead(4);
      accelbuffZ[5]=analogRead(5);
      accelbuffZ[6]=analogRead(6);
      accelbuffZ[7]=analogRead(7)+4500;
//      accelbuffZ[0]=analogRead(1);
//      accelbuffZ[1]=analogRead(0);
//      accelbuffZ[2]=analogRead(4);
//      accelbuffZ[3]=analogRead(3);
//      accelbuffZ[4]=3000;
//      accelbuffZ[5]=3000;
//      accelbuffZ[6]=3000;
//      accelbuffZ[7]=3000+4500;
      SerialUSB.write((uint8_t *)accelbuffZ,16);
//      Serial.write((uint8_t *)accelbuffZ,16);
  }

}
