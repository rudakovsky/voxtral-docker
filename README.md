# voxtral-docker
Mistral Voxtral within a docker container

Get HF token:
https://huggingface.co/settings/tokens → New token (Read)

Download local model:
download.sh

Build:
docker build -t voxtral-mini-local .

Put wav in the same directory:
your.wav

Run:
docker run --rm -v "$(pwd)":/audio -v "$(pwd)/models":/app/models voxtral-mini-local /audio/your.wav 2>&1 | tee debug.log.


