#!/bin/bash
sudo rm -f /etc/resolv.conf
echo "nameserver 8.8.8.8" | sudo tee /etc/resolv.conf
echo "nameserver 1.1.1.1" | sudo tee -a /etc/resolv.conf
echo "nameserver 8.8.4.4" | sudo tee -a /etc/resolv.conf
sudo chattr +i /etc/resolv.conf
echo "============================================="
echo "DA FIX XONG LOI DNS! BAN HAY CHAY APT UPDATE LAI!"
echo "============================================="
