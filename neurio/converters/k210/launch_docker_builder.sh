
# Clone repository and launch docker
git clone https://github.com/kendryte/canmv
cd canmv
git submodule update --recursive --init
git checkout feat/kpu
git pull # to check it is up to date
cd tools/docker
sh start.sh # starts docker
cd ../../ # back to root
docker exec -it k210_build bash # opens a docker interactive console
