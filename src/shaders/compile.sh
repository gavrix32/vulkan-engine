#!/bin/bash

SHADER_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$SHADER_DIR/spirv"

mkdir -p "$OUTPUT_DIR"

echo "Compiling shaders..."

for SHADER in pbr light blit; do
    SRC="$SHADER_DIR/$SHADER.slang"
    DST="$OUTPUT_DIR/$SHADER.spv"

    if [[ ! -f "$DST" || "$SRC" -nt "$DST" ]]; then
        echo "- $SHADER"
        slangc -I "$SHADER_DIR" "$SRC" -target spirv -profile spirv_1_4 -matrix-layout-column-major -o "$DST"

        if [[ $? -ne 0 ]]; then
            echo "Failed to compile $SHADER"
            exit 1
        fi
    fi
done

echo "Done"