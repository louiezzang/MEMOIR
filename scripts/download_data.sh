#!/bin/bash
# Download datasets for MEMOIR experiments
# Usage: bash scripts/download_data.sh [amazon|mind|movielens|all] [full]
#   The optional "full" argument (amazon only) downloads all 32 Amazon Reviews 2023
#   categories instead of just Electronics + Clothing_Shoes_and_Jewelry.

set -e

DATA_DIR="./data/raw"
mkdir -p "$DATA_DIR"

download_amazon() {
    echo "=== Downloading Amazon Reviews 2023 ==="
    echo "Source: https://amazon-reviews-2023.github.io/"
    AMAZON_DIR="$DATA_DIR/amazon"
    mkdir -p "$AMAZON_DIR"

    # Using the McAuley Lab Amazon Reviews 2023 dataset
    if [ "$1" == "full" ]; then
        # All 32 official categories (per amazon-reviews-2023.github.io stats table)
        CATEGORIES=(
            "All_Beauty" "Amazon_Fashion" "Appliances" "Arts_Crafts_and_Sewing"
            "Automotive" "Baby_Products" "Beauty_and_Personal_Care" "Books"
            "CDs_and_Vinyl" "Cell_Phones_and_Accessories" "Clothing_Shoes_and_Jewelry"
            "Digital_Music" "Electronics" "Gift_Cards" "Grocery_and_Gourmet_Food"
            "Handmade_Products" "Health_and_Household" "Health_and_Personal_Care"
            "Home_and_Kitchen" "Industrial_and_Scientific" "Kindle_Store"
            "Magazine_Subscriptions" "Movies_and_TV" "Musical_Instruments"
            "Office_Products" "Patio_Lawn_and_Garden" "Pet_Supplies" "Software"
            "Sports_and_Outdoors" "Tools_and_Home_Improvement" "Toys_and_Games"
            "Video_Games"
        )
    else
        CATEGORIES=("Electronics" "Clothing_Shoes_and_Jewelry")
    fi
    META_DIR="$AMAZON_DIR/raw/meta_categories"
    mkdir -p "$META_DIR"

    for CAT in "${CATEGORIES[@]}"; do
        REVIEW_URL="https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/${CAT}.jsonl.gz"
        META_URL="https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_${CAT}.jsonl.gz"
        OUT="$AMAZON_DIR/${CAT}.jsonl.gz"
        META_OUT="$META_DIR/meta_${CAT}.jsonl.gz"

        if [ ! -f "$AMAZON_DIR/${CAT}.jsonl" ]; then
            echo "Downloading $CAT reviews..."
            if command -v wget >/dev/null 2>&1; then
                wget -q --show-progress --timeout=30 --tries=5 "$REVIEW_URL" -O "$OUT"
            else
                # Use curl with resume and retry logic for large files
                MAX_RETRIES=5
                RETRY_WAIT=5
                for i in $(seq 1 $MAX_RETRIES); do
                    echo "Attempt $i/$MAX_RETRIES..."
                    if curl -L -C - --retry 3 --retry-delay $RETRY_WAIT --connect-timeout 30 "$REVIEW_URL" -o "$OUT"; then
                        break
                    fi
                    if [ $i -eq $MAX_RETRIES ]; then
                        echo "Failed to download after $MAX_RETRIES attempts"
                        exit 1
                    fi
                    sleep $RETRY_WAIT
                done
            fi
            gunzip -c "$OUT" > "$AMAZON_DIR/${CAT}.jsonl"
            rm -f "$OUT"
            echo "$CAT reviews done."
        else
            echo "$CAT reviews already exists, skipping."
        fi

        if [ ! -f "$META_DIR/meta_${CAT}.jsonl" ]; then
            echo "Downloading $CAT meta categories..."
            if command -v wget >/dev/null 2>&1; then
                wget -q --show-progress --timeout=30 --tries=5 "$META_URL" -O "$META_OUT"
            else
                # Use curl with resume and retry logic for large files
                MAX_RETRIES=5
                RETRY_WAIT=5
                for i in $(seq 1 $MAX_RETRIES); do
                    echo "Attempt $i/$MAX_RETRIES..."
                    if curl -L -C - --retry 3 --retry-delay $RETRY_WAIT --connect-timeout 30 "$META_URL" -o "$META_OUT"; then
                        break
                    fi
                    if [ $i -eq $MAX_RETRIES ]; then
                        echo "Failed to download after $MAX_RETRIES attempts"
                        exit 1
                    fi
                    sleep $RETRY_WAIT
                done
            fi
            gunzip -c "$META_OUT" > "$META_DIR/meta_${CAT}.jsonl"
            rm -f "$META_OUT"
            echo "$CAT meta categories done."
        else
            echo "$CAT meta categories already exists, skipping."
        fi
    done
    echo "Amazon Reviews download complete."
}

download_mind() {
    echo "=== Downloading MIND Dataset ==="
    echo "Source: https://msnews.github.io/"
    MIND_DIR="$DATA_DIR/mind"
    mkdir -p "$MIND_DIR"

    # MIND Small (for development)
    if [ ! -f "$MIND_DIR/MINDsmall_train.zip" ]; then
        echo "Downloading MIND Small (train)..."
        wget -q --show-progress "https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip" -O "$MIND_DIR/MINDsmall_train.zip" \
            || curl -L "https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip" -o "$MIND_DIR/MINDsmall_train.zip"
        mkdir -p "$MIND_DIR/MINDsmall"
        unzip -q "$MIND_DIR/MINDsmall_train.zip" -d "$MIND_DIR/MINDsmall"
        rm "$MIND_DIR/MINDsmall_train.zip"

        echo "Downloading MIND Small (dev)..."
        wget -q --show-progress "https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip" -O "$MIND_DIR/MINDsmall_dev.zip" \
            || curl -L "https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip" -o "$MIND_DIR/MINDsmall_dev.zip"
        mkdir -p "$MIND_DIR/MINDsmall"
        unzip -q "$MIND_DIR/MINDsmall_dev.zip" -d "$MIND_DIR/MINDsmall"
        rm "$MIND_DIR/MINDsmall_dev.zip"
    else
        echo "MIND Small already exists, skipping."
    fi
    echo "MIND download complete."
}

download_movielens() {
    echo "=== Downloading MovieLens (ml-latest-small) ==="
    echo "Source: https://grouplens.org/datasets/movielens/"
    ML_DIR="$DATA_DIR/movielens"
    mkdir -p "$ML_DIR"

    if [ ! -f "$ML_DIR/ml-latest-small.zip" ]; then
        echo "Downloading MovieLens ml-latest-small..."
        wget -q --show-progress "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip" -O "$ML_DIR/ml-latest-small.zip" \
            || curl -L "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip" -o "$ML_DIR/ml-latest-small.zip"
        unzip -q "$ML_DIR/ml-latest-small.zip" -d "$ML_DIR"
        rm "$ML_DIR/ml-latest-small.zip"
    else
        echo "MovieLens ml-latest-small already exists, skipping."
    fi
    echo "MovieLens download complete."
}

case "${1:-all}" in
    amazon)    download_amazon "$2" ;;
    mind)      download_mind ;;
    movielens) download_movielens ;;
    all)
        download_amazon
        download_mind
        download_movielens
        ;;
    *)
        echo "Usage: $0 [amazon|mind|movielens|all]"
        exit 1
        ;;
esac

echo ""
echo "=== All downloads complete ==="
echo "Data directory: $DATA_DIR"
ls -la "$DATA_DIR"/*/
