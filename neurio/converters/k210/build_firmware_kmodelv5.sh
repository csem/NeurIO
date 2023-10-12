
#In docker, configure and build
git branch # correct - feat/kpu

# make sure the kendryte toolchain is up-to-date in docker.
wget https://github.com/kendryte/kendryte-gnu-toolchain/releases/download/v8.2.0-20190409/kendryte-toolchain-ubuntu-amd64-8.2.0-20190409.tar.xz
tar -Jxvf kendryte-toolchain-ubuntu-amd64-8.2.0-20190409.tar.xz -C /opt
ls /opt/kendryte-toolchain/bin

# update the NNCase files to 1.8.0
wget https://github.com/kendryte/nncase/releases/download/v1.8.0/nncaseruntime-k210.zip
unzip nncaseruntime-k210.zip -d nncaseruntime-k210
cp -r nncaseruntime-k210/include components/kendryte_sdk/kendryte-standalone-sdk/lib/nncase/v1/
cp -r nncaseruntime-k210/lib components/kendryte_sdk/kendryte-standalone-sdk/lib/nncase/v1/

# update the NNCase files to 1.9.0


# clean files
rm kendryte-toolchain-ubuntu-amd64-8.2.0-20190409.tar.xz
rm nncaseruntime-k210.zip
rm nncaseruntime-k210


# go to M1 project folder.
cd projects/canmv_dock
# clean project
python project.py clean
python project.py distclean
# configure, can also be done manually by modifying config_defaults.mk file. Set KModel V5 support to True.
#python project.py menuconfig
# set manually
sed -i 's/CONFIG_CANMV_ENABLE_KMODEL_V4=y/# CONFIG_CANMV_ENABLE_KMODEL_V4 is not set\nCONFIG_CANMV_ENABLE_KMODEL_V5=y/' config_defaults.mk

# then compile
python project.py build --config_file config_defaults.mk

# copy and burn firmware using CanMV IDE.