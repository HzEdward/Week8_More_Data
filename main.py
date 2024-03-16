# execute main function in pickwhitelist.py for 100 times

import pick_whitelist

def main():
    count = 0
    for i in range(50):
        count += 1
        pick_whitelist.main()

    print(count)

if __name__ == "__main__":
    main()





    

