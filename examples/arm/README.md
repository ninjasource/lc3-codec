# Lc3-codec Arm emulator example

This example runs on an 168mhz arm cortex-m4f mcu emulated in QEMU
You need at least qemu version 7.0.0 installed so that you can use the netduinoplus2 mcu which has support for hardware float (FPU)

At the time of writing the virtual chip above decodes one audio frame in `3045 microseconds`
and a real 64mhz cortex-m4f nordic 52840 mcu decodes the same frame in `5432 microseconds` to give you a comparison.

The virtual chip timings are not host wall clock times but calculated by the number of virtual
cpu cycles so they are stable.

Not sure why the virtual chip is not faster but it's probably some clock scaling thing.

To run
```
cd ./example/arm

cargo run --release
```

If you don't have the latest qemu (I had to compile my own) then you can change the machine in the .cargo/config.toml file
