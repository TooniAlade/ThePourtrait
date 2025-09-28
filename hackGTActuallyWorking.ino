#include <Adafruit_NeoPixel.h>
#include "Adafruit_seesaw.h"
#include <seesaw_neopixel.h>
#include <Wire.h>

#define LED_PIN 8
#define LED_COUNT 144
#define OUTER_PIN 10
#define OUTER_COUNT 24
#define INNER_PIN 11
#define INNER_COUNT 16
#define BRIGHTNESS 50  // NeoPixel brightness

#define SS_SWITCH 24
#define SS_NEOPIX 6
#define SEESAW_ADDR 0x36

Adafruit_seesaw ss;
seesaw_NeoPixel sspixel = seesaw_NeoPixel(1, SS_NEOPIX, NEO_GRBW + NEO_KHZ800);

Adafruit_NeoPixel strip(LED_COUNT, LED_PIN, NEO_GRB + NEO_KHZ800);
Adafruit_NeoPixel outer(OUTER_COUNT, OUTER_PIN, NEO_RGBW + NEO_KHZ800);
Adafruit_NeoPixel inner(INNER_COUNT, INNER_PIN, NEO_RGB + NEO_KHZ800);

int32_t encoder_position;
int wrappedValue;
int currIteration = 0;
bool prevButtonState = false;

// Inner strip (NEO_RGB) – unchanged
uint32_t innerColors[24] = {
  0xFF0000, 0xFF3F00, 0xFF7F00, 0xFFBF00, 0xFFFF00, 0xBFFF00,
  0x7FFF00, 0x3FFF00, 0x00FF00, 0x00FF3F, 0x00FF7F, 0x00FFBF,
  0x00FFFF, 0x00BFFF, 0x007FFF, 0x003FFF, 0x0000FF, 0x3F00FF,
  0x7F00FF, 0xBF00FF, 0xFF00FF, 0xFF00BF, 0xFF007F, 0xFF003F
};

// Main strip (NEO_GRB) – R and G swapped
uint32_t stripColors[24] = {
  0x00FF00, 0x3FFF00, 0x7FFF00, 0xBFFF00, 0xFFFF00, 0xFFBF00,
  0xFF7F00, 0xFF3F00, 0xFF0000, 0xFF003F, 0xFF007F, 0xFF00BF,
  0xFF00FF, 0xBF00FF, 0x7F00FF, 0x3F00FF, 0x0000FF, 0x003FFF,
  0x007FFF, 0x00BFFF, 0x00FFFF, 0x00FFBF, 0x00FF7F, 0x00FF3F
};

uint32_t chosenColors[6];


void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);

  // Initialize seesaw
  if (!ss.begin(SEESAW_ADDR) || !sspixel.begin(SEESAW_ADDR)) {
    // Serial.println("Couldn't find seesaw!");
    while (1) delay(10);
  }

  ss.pinMode(SS_SWITCH, INPUT_PULLUP);
  encoder_position = ss.getEncoderPosition();
  ss.setGPIOInterrupts((uint32_t)1 << SS_SWITCH, 1);
  ss.enableEncoderInterrupt();

  // Initialize NeoPixels
  strip.begin();
  strip.show();
  strip.setBrightness(BRIGHTNESS);
  outer.begin();
  outer.show();
  outer.setBrightness(BRIGHTNESS);
  inner.begin();
  inner.clear();
  inner.show();
  inner.setBrightness(BRIGHTNESS);

  // Fill outer and inner with initial colors
  for (int i = 0; i < OUTER_COUNT; i++) {
    outer.setPixelColor(i, innerColors[i % 24]);
  }
  outer.show();
}

void loop() {

  if (currIteration < 6) {
  bool currButtonState = !(ss.digitalRead(SS_SWITCH));

  if ((currButtonState) && (!prevButtonState)) {
    // Serial.println("pressed");
  }

  if ((!currButtonState) && (prevButtonState)) {
    // Serial.println("released");
    // Serial.print("wrapped position is ");
    // Serial.println(wrappedValue);

    // Fill main strip with the same color as inner
    uint32_t fillColor = stripColors[wrappedValue];
    for (int i = currIteration * 24; i < currIteration * 24 + 24; i++) {
      strip.setPixelColor(i, fillColor);
    }
    strip.show();
    chosenColors[currIteration] = stripColors[wrappedValue];
    currIteration++;
  }

  prevButtonState = currButtonState;

  // Encoder logic
  int32_t new_position = ss.getEncoderPosition();
  if (encoder_position != new_position) {
    wrappedValue = new_position % 24;
    if (wrappedValue < 0) wrappedValue += 24;

    // Serial.print("wrapped position is ");
    // Serial.println(wrappedValue);

    // Update inner strip with selected color
    inner.fill(innerColors[wrappedValue]);
    inner.show();

    encoder_position = new_position;
  }

  delay(10);
  }
  if (currIteration == 6) {
    String finalString = "";


    for (int i = 0; i < 6; i++) 
    { Serial.println(chosenColors[i]); 
      // Serial.println(chosenTimes[])
    }
    currIteration++;
  }
}
