#!/bin/bash
# Generate traffic to the Heart Disease API for monitoring demo.
# Usage: ./scripts/generate_traffic.sh [API_URL] [NUM_REQUESTS]

API_URL=${1:-"http://34.60.20.112"}
NUM_REQUESTS=${2:-50}

echo "Sending $NUM_REQUESTS requests to $API_URL/predict..."

for i in $(seq 1 $NUM_REQUESTS); do
  # Randomize some features for variety
  AGE=$((40 + RANDOM % 40))
  SEX=$((RANDOM % 2))
  CP=$((RANDOM % 4))
  TRESTBPS=$((100 + RANDOM % 80))
  CHOL=$((150 + RANDOM % 200))
  FBS=$((RANDOM % 2))
  RESTECG=$((RANDOM % 3))
  THALACH=$((100 + RANDOM % 100))
  EXANG=$((RANDOM % 2))
  OLDPEAK=$(echo "scale=1; $((RANDOM % 50)) / 10" | bc)
  SLOPE=$((RANDOM % 3))
  CA=$((RANDOM % 4))
  THAL=$((RANDOM % 4))

  curl -s -X POST "$API_URL/predict" \
    -H "Content-Type: application/json" \
    -d "{\"age\": $AGE, \"sex\": $SEX, \"cp\": $CP, \"trestbps\": $TRESTBPS, \"chol\": $CHOL, \"fbs\": $FBS, \"restecg\": $RESTECG, \"thalach\": $THALACH, \"exang\": $EXANG, \"oldpeak\": $OLDPEAK, \"slope\": $SLOPE, \"ca\": $CA, \"thal\": $THAL}" \
    -o /dev/null -w "Request $i: HTTP %{http_code} in %{time_total}s\n"

  sleep 0.5
done

echo "Done. Check Grafana dashboard for metrics."
