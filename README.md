# lc3-codec

Low Complexity Communication Codec. This is an implementation of the the Bluetooth(r) LC3 Audio Codec revision 1.0 (released on 2020-09-15) targeting `no_std` environments. 

This is not currently approved or verified in any formal way other than my own testing against a codec that has been validated. Both encoding and decoding are working for some music I have thrown at it. The music files have not been included in this repo for copyright reasons.

To start take a look at the `lc3_decoder.rs` and `lc3_encoder.rs` files.

## Introduction

The purpose of this codebase is to show how a modern audio codec works. It was written to run on an embedded mcu so the API may seem a little awkward because you have to pass in preallocated memory. My background is not in signal processing and I wanted to create a codebase that someone like me could read and understand. This is why I try to avoid the shorthand variable and method names you may see in similar implementations. The excessive commenting is for my benefit and for those with less experience in signal processing. 

The codebase is very much a work in progress and I am actively working on performance enhancements and general simplification of anything that looks confusing.

## Encoder Usage

### On a system with an allocator

```toml
# Cargo.toml
lc3-codec = { version = "0.2" }
```

```rust
// setup the encoder
let num_channels = 1;
let sampling_frequency = SamplingFrequency::Hz48000;
let frame_duration = FrameDuration::TenMs;
let (integer_length, scaler_length, complex_length) =
    Lc3Encoder::calc_working_buffer_lengths(num_channels, frame_duration, sampling_frequency);
let mut integer_buf = vec![0; integer_length];
let mut scaler_buf = vec![0.0; scaler_length];
let mut complex_buf = vec![Complex::default(); complex_length];
let mut encoder = Lc3Encoder::new(
    num_channels, frame_duration, sampling_frequency, &mut integer_buf, &mut scaler_buf, &mut complex_buf,
);

// encode a frame of audio on channel 0
let samples_in: Vec<i16> = vec![0; 480];
let mut buf_out: Vec<u8> = vec![0; 150];
encoder.encode_frame(0, &samples_in, &mut buf_out).unwrap();
```

### In a `no_std` env (no allocator)

```toml
# Cargo.toml
lc3-codec = { version = "0.2", default-features = false }
```

```rust
// setup the encoder statically
const NUM_CH: usize = 1;
const DURATION: FrameDuration = FrameDuration::TenMs;
const FREQ: SamplingFrequency = SamplingFrequency::Hz48000;
const BUF_LENGTHS: (usize, usize, usize) = Lc3Encoder::<NUM_CH>::calc_working_buffer_lengths(DURATION, FREQ);
let mut integer_buf = [0; BUF_LENGTHS.0];
let mut scaler_buf = [0.0; BUF_LENGTHS.1];
let mut complex_buf = [Complex::default(); BUF_LENGTHS.2];
let mut encoder =
    Lc3Encoder::<NUM_CH>::new(DURATION, FREQ, &mut integer_buf, &mut scaler_buf, &mut complex_buf);

// encode a frame of audio on channel 0
let samples_in: [i16; 480] = [0; 480];
let mut buf_out: [u8; 150] = [0; 150];
encoder.encode_frame(0, &samples_in, &mut buf_out).unwrap();
```

## Decoder Usage

### On a system with an allocator

```toml
# Cargo.toml
lc3-codec = { version = "0.2" }
```

```rust
// setup decoder
let num_channels = 1;
let sampling_frequency = SamplingFrequency::Hz48000;
let frame_duration = FrameDuration::TenMs;
let (scaler_length, complex_length) =
    Lc3Decoder::calc_working_buffer_lengths(num_channels, frame_duration, sampling_frequency);
let mut scaler_buf = vec![0.0; scaler_length];
let mut complex_buf = vec![Complex::default(); complex_length];
let mut decoder = Lc3Decoder::new(
    num_channels, frame_duration, sampling_frequency, &mut scaler_buf, &mut complex_buf,
);

// decode 16 bit audio on channel 0
let buf_in: Vec<u8> = vec![0; 150];
let mut samples_out: Vec<i16> = vec![0; 480];
decoder.decode_frame(16, 0, &buf_in, &mut samples_out).unwrap();
```

### In a `no_std` env (no allocator)

```toml
# Cargo.toml
lc3-codec = { version = "0.2", default-features = false }
```

```rust
// setup decoder
const NUM_CH: usize = 1;
const FREQ: SamplingFrequency = SamplingFrequency::Hz48000;
const DURATION: FrameDuration = FrameDuration::TenMs;
const SCALER_COMPLEX_LENS: (usize, usize) = Lc3Decoder::calc_working_buffer_lengths(1, DURATION, FREQ);
let mut scaler_buf = [0.0; SCALER_COMPLEX_LENS.0];
let mut complex_buf = [Complex::default(); SCALER_COMPLEX_LENS.1];
let mut decoder = Lc3Decoder::new(NUM_CH, DURATION, FREQ, &mut scaler_buf, &mut complex_buf);

// decode 16 bit audio on channel 0 
let mut samples_out = [0; 480];
let buf_in: [u8; 150] = [0; 150];
decoder.decode_frame(16, 0, &buf_in, &mut samples_out).unwrap();
```

## Performance and system requirements

Here are some performance benchmarks at the time of writing.

On a PC:
The decoder can decode `60 minutes` of compressed audio in `5 seconds` on one core of a `i7-6700K` CPU @ 4.00GHz

On a microcontroller:
The decoder can decode `10 ms` of audio (one frame) in `5.4 ms` on a Nordic `nrf52840` Cortex M4F MCU running at 72mhz

The decoder uses `27,564` bytes of ram for its working buffers

## Unit tests

The unit tests in place right now are primarily here to facilitate refactoring by helping me identify changes with material effects to the input and output data. Basically I don't want to break things and this helps with that. More granular and useful tests will come as soon as the codebase stabilizes a little.

## License

``` 
Copyright 2022 David Haig

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at:

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```