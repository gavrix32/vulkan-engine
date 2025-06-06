#!/bin/bash

SHADER_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$SHADER_DIR/spirv"

slangc "$SHADER_DIR/vertex.slang" -profile glsl_450 -target spirv -o "$OUTPUT_DIR/vertex.spv"
slangc "$SHADER_DIR/fragment.slang" -profile glsl_450 -target spirv -o "$OUTPUT_DIR/fragment.spv"
