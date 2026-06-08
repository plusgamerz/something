import socket, time

PORT = 12345

s = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
s.settimeout(3)
s.bind(('0.0.0.0',PORT))

print("Starting Client!")
print("Listening...")

try:
    while True:
        while True:
            try:
                message, addr  = s.recvfrom(1024)
                print(f"IP: {addr[0]}, PORT: {addr[1]}",end=', ')
                print("Recieved: "+message.decode())
                break
            except socket.timeout:
                 time.sleep(1)
                 continue
            except Exception as e:
                print("Error: "+e)
                time.sleep(3)
except KeyboardInterrupt:
        print("Closing Client!")
finally:
    s.close()
    quit()