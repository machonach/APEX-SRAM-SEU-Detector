This is our SRAM SEU Detector for APEX! It consists of multiple parts: a Raspberry Pi Zero 2 W, two Pico W microcontrollers, a GPS module, a pressure + temperature sensor, and SRAM chips. What this project does is collect data on SEUs (Single-Event Upsets) in the SRAM, such as with bit flips. We then collect the frequency of the SEUs from the ground and up to 30 km in the air along with the pressure, altitude, temperature, and location in order to see when it is optimal for an SEU to occur. We then use this data to train a machine learning model to predict when SEU events would occur in real time.

<picture>
 <img alt="Main Component" src="https://github.com/machonach/APEX-SRAM-SEU-Detector/blob/main/images/IMG_5444.png" width="300">
</picture>

<picture>
  <img alt="SRAM" src="https://github.com/machonach/APEX-SRAM-SEU-Detector/blob/main/images/IMG_5447.png" width="300">
</picture>

<picture>
  <img alt="Synthetic Data Analysis" src="https://github.com/machonach/APEX-SRAM-SEU-Detector/blob/main/images/seu_synthetic_analysis_results.png" width="604">
</picture>

WIP

Members:
Yasashvee Karthi, Alexander Jameson, Om Ghosh, Atrey Iyer
