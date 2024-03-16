# execute main function in pickwhitelist.py for 100 times

import only_whitelist

def main():
    count = 0
    for i in range(50):
        count += 1
        only_whitelist.main()

    print(count)

if __name__ == "__main__":
    main()





    

