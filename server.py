import socket, time

IP = "255.255.255.255"
PORT = 12345

s = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET,socket.SO_BROADCAST,1)
s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)

print("Server Started!")

try:
    while True:
        message = input("Broadcast message: ").encode('utf-8')
        s.sendto(message,(IP,PORT))
except KeyboardInterrupt:
    print("Closing Server!")
finally:
    s.close()
    quit()
