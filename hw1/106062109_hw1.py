import requests
from bs4 import BeautifulSoup
import re

def read_des(input_file):
    des = []
    file = open(input_file, 'r')
    lines = file.read().splitlines()
    return lines

def next_des(first_des, file, url):
    count = 1
    to_who = [first_des]
    info = {}
    fp = open("106062109_hw1_output.txt", "w")
        
    def chain(to_who):
        for who in to_who[:-1]:
            fp.write("%s -> " % who)
        fp.write("%s\n" % to_who[-1])
        fp.write("--------------------------------------------------------------------------\n")
    
    while url != "":  
        info.clear()
        resp = requests.get(url)
        resp.encoding = 'utf-8'
        if resp.status_code==200: #normal
            soup = BeautifulSoup(resp.text, 'html.parser')
            detail = soup.find_all('span', {'class': 'sc-1ryi78w-0 bFGdFC sc-16b9dsl-1 iIOvXh u3ufsr-0 gXDEBk'})
            divs = soup.find_all('div', {'class': 'sc-1fp9csv-0 gkLWFf', 'direction': "vertical"})

            fp.write("Nonce: %s \n" % detail[1].text.strip())
            fp.write("Number of Transactions: %s \n" % detail[2].text.strip())
            fp.write("Final Balance: %s \n" % detail[3].text.strip())
            fp.write("Total Sent: %s \n" % detail[4].text.strip())
            fp.write("Total Received: %s \n" % detail[5].text.strip())
            fp.write("Total Fees: %s \n" % detail[6].text.strip())

            num = len(divs) - 1
            while num >= 0:
                found = divs[num]
                found = found.get_text()
                found = re.split('Hash|Date|From|To|Fee|Amount',found) # cool 
                if found[6][0] == "-":
                    info = {
                        'Date': found[2],
                        'To': found[4],
                        'Amount': found[6] }
                    break
                num -= 1

            if len(info) == 0: 
                if len(file) != 0:
                    fp.write("--------------------------------------------------------------------------\n")
                    chain(to_who)
                    url = "https://www.blockchain.com/eth/address/" + file[0] + "?view=standard"
                    count = 1
                    to_who = []
                    to_who.append(file[0])
                    file.pop(0)
                else: # no out turn
                    fp.write("--------------------------------------------------------------------------\n")
                    chain(to_who)
                    break
            else:
                fp.write("Date: %s \n" % info['Date'])
                fp.write("To: %s \n" % info['To'])
                fp.write("Amount: %s \n" % info['Amount'])
                fp.write("--------------------------------------------------------------------------\n")
                url = "https://www.blockchain.com/eth/address/" + info['To'] + "?view=standard"
                count += 1

                if count == 5: 
                    chain(to_who)
                    if len(file) == 0: # no more txt
                        break
                    else:
                        url = "https://www.blockchain.com/eth/address/" + file[0] + "?view=standard"
                        count = 1
                        to_who = []
                        to_who.append(file[0])
                        file.pop(0)
                else:
                    to_who.append(info['To'])    
    
    fp.close()

def main():
    des = read_des("input_hw1.txt")
    #print(des)
    first_url = "https://www.blockchain.com/eth/address/" + des[0] + "?view=standard"
    next_des(des[0],des[1:], first_url)

main()

