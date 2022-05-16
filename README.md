# lc3-codec

Low Complexity Communication Codec. This is an implementation of the the Bluetooth(r) LC3 Audio Codec revision 1.0 (released on 2020-09-15) targeting `no_std` environments. 

This is not currently approved or verified in any formal way other than my own testing against a codec that has been validated. Both encoding and decoding are working for some music I have thrown at it. The music files have not been included in this repo for copyright reasons.

To start take a look at the `lc3_decoder.rs` and `lc3_encoder.rs` files.

## Introduction

The purpose of this codebase is to show how a modern audio codec works. It was written to run on an embedded mcu so the API may seem a little awkward because you have to pass in preallocated memory. My background is not in signal processing and I wanted to create a codebase that someone like me could read and understand. This is why I try to avoid the shorthand variable and method names you may see in similar implementations. The excessive commenting is for my benefit and for those with less experience in signal processing. 

The codebase is very much a work in progress and I am actively working on performance enhancements and general simplification of anything that looks confusing.

## Unit tests

The unit tests in place right now are primarily here to facilitate refactoring by helping me identify changes with material effects to the input and output data. Basically I don't want to break things and this helps with that. More granular and useful tests will come as soon as the codebase stabilizes a little.

# License

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