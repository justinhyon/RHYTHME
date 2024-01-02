#include <Wire.h>
#define MCP4725_ADDR 0x62
#include <Adafruit_MCP4725.h>

Adafruit_MCP4725 dac;

const int ledPin = 6;

const float freq = 10.;
float lenSig = .5;

const int HANDSHAKE = 0;
const int OFF = 2;
const int SIGNAL = 3;

// Initially, only send data upon request
int daqMode = OFF;

// Default time between data points in ms
int daqDelay = 10;


// Keep track of last data acquistion for delays
unsigned long timeOfLastDAQ = 0;

bool firstLoop = true;
unsigned long firstTime = 0;

int count = 0;

void write12BitI2C(int x) {
  /*
   * Write a 12-bit integer out to I2C.
   */
//  Serial.println("test2");
  Wire.beginTransmission(MCP4725_ADDR);

  Wire.write(64);            // cmd to update the DAC
  Wire.write(x >> 4);        // the 8 most significant bits...
  Wire.write((x % 16) << 4); // the 4 least significant bits...

  Wire.endTransmission();
//  Serial.println("test5");
}

void setup() {
  // Initialize serial communication
  Serial.begin(115200);

  Wire.begin();
//  dac.begin(MCP4725_ADDR);
  pinMode(ledPin, OUTPUT);
  
}


void loop() {
  // If we're streaming

  if (daqMode == SIGNAL) {
    // Turn the LED on, and then wait a quarter second
//    digitalWrite(ledPin, HIGH);
//    delay(250);
//  
//    // Turn the LED off, and then wait a quarter second
//    digitalWrite(ledPin, LOW);
//    delay(250);
    unsigned long currTime = millis();
    if (firstLoop == true){
      firstTime = currTime;
      firstLoop = false;
    }
    
    if (currTime - timeOfLastDAQ >= daqDelay) {
      uint16_t x = (uint16_t)(4095 * (1 + sin(2 * PI * freq * currTime / 1000.0)) / 2.0);

      write12BitI2C(x);

      timeOfLastDAQ = currTime;
      count = count + 1;
      
    }

    if (currTime >= firstTime + (lenSig * 1000)){
      write12BitI2C(0);
      daqMode = OFF;
      firstLoop = true;
    }
  }

  // Check if data has been sent to Arduino and respond accordingly
  if (Serial.available() > 0) {
    // Read in request
    int inByte = Serial.read();

    // If data is requested, fetch it and write it, or handshake
    switch(inByte) {
      case OFF:
        daqMode = OFF;
        break;
      case SIGNAL:
        daqMode = SIGNAL;
        
        break;
      case HANDSHAKE:
        if (Serial.availableForWrite()) {
          Serial.println("Message received.");
        }
        break;
    }
  }
}
