#include <WiFi.h>
#include <WiFiUdp.h>

#include "FastIMU.h"

// for data collection
#define IMU_ADDRESS 0x68
MPU6500 IMU;
calData calib = { 0 };  
AccelData accelData;    
GyroData gyroData;

const int id = 1;

const char* ssid = "";
const char* password = "";
const int udp_port = 12345;

WiFiUDP Udp;

void setup() {
  Wire.begin();
  Wire.setClock(400000);
  Serial.begin(115200);

  int err = IMU.init(calib, IMU_ADDRESS);
  delay(5000);
  Serial.println("Keep IMU level.");
  delay(5000);
  IMU.calibrateAccelGyro(&calib);
  Serial.println("Calibration done!");
  delay(5000);
  IMU.init(calib, IMU_ADDRESS);

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }
  Serial.println("Connected to WiFi");

  Udp.begin(udp_port);
}

String v[26];

void loop() {
  int packetSize = Udp.parsePacket();
  if (packetSize) {
    Serial.println("Received something!!!");
    char packetBuffer[255];
    int len = Udp.read(packetBuffer, 255);
    if (len > 0) {
      
      for(int i = 1; i <= 25; i++){
        v[i] = "";
      }

      for(int i = 1; i <= 500; i++){
        int ind = (i - 1) / 25 + 1;

        //Serial.println(millis());
        IMU.update();
        IMU.getAccel(&accelData);
        IMU.getGyro(&gyroData);

        String s = "";
        s += String(id);
        s += ",";
        s += String(ind);
        s += ",";
        s += millis();
        s += ",";
        s += String(accelData.accelX);
        s += ",";
        s += String(accelData.accelY);
        s += ",";
        s += String(accelData.accelZ);
        s += ",";
        s += String(gyroData.gyroX);
        s += ",";
        s += String(gyroData.gyroY);
        s += ",";
        s += String(gyroData.gyroZ);

        if(i % 25 != 0){
          s += ";";
        }

        v[ind] += s;

        delay(4);
      }

      for(int i = 1; i <= 20; i++){
        Serial.println(i);
        Udp.beginPacket(Udp.remoteIP(), 12346);  
        Udp.print(v[i]);
        Udp.endPacket();
        delay(4);
      }
    }
    
  }
}
