#!/bin/bash

SHADER_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$SHADER_DIR/spirv"

if [[ "$SHADER_DIR/vertex.slang" -ot "$OUTPUT_DIR/vertex.spv" && \
      "$SHADER_DIR/fragment.slang" -ot "$OUTPUT_DIR/fragment.spv" ]]; then
    exit 0
fi

slangc "$SHADER_DIR/vertex.slang" -target spirv -profile spirv_1_4 -o "$OUTPUT_DIR/vertex.spv"
slangc "$SHADER_DIR/fragment.slang" -target spirv -profile spirv_1_4 -o "$OUTPUT_DIR/fragment.spv"
