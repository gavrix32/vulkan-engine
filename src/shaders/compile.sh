#!/bin/bash

SHADER_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$SHADER_DIR/spirv"

slangc "$SHADER_DIR/vertex.slang" -target spirv -profile spirv_1_4 -o "$OUTPUT_DIR/vertex.spv"
slangc "$SHADER_DIR/fragment.slang" -target spirv -profile spirv_1_4 -o "$OUTPUT_DIR/fragment.spv"
