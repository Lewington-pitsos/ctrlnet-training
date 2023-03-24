echo "pod started"

mkdir -p ~/.ssh
chmod 700 ~/.ssh
cd ~/.ssh
echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPL01JFuJPPamDUFNd7YLU4owinDuDI459HIpz+Xg5Ya lewingtonpitsos@gmail.com" >> authorized_keys
chmod 700 -R ~/.ssh
cd /
service ssh start

sleep infinity