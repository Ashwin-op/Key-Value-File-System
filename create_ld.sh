# unmount the loopback device
#umount /mnt/loopback

# remove the loopback device
#losetup -d /dev/loop0

# remove the file if exists
rm -f loopback.img

# first create a file
dd if=/dev/zero of=loopback.img bs=1M count=100

# create a loopback device
#losetup /dev/loop0 loopback.img

# check the loopback device
#losetup -a

# format the loopback device
#mkfs.ext4 /dev/loop0

# mount the loopback device
#mkdir /mnt/loopback
#mount /dev/loop0 /mnt/loopback

# check the loopback device
#df -h
