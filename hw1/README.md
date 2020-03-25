# Data Science HW1

### Goal: Crawling down the ETH data 



#### Using tools
  - python3
  - Requests
  - BeautifulSoup


---

![](https://i.imgur.com/b4gGMMi.png)

## Input:

  - A txt file saved address destinations: input_hw1.txt
  - One address for each line

## Steps:


1. Use requests to get website contents in `https://www.blockchain.com/eth/address/目標地址?view=standard`

https://www.blockchain.com/eth/address/0x03f034fb47965123ea4148e3147e2cfdc5b1f7a5?view=standard

2. Use BeautifulSoup to analyze and grab useful data

## Explanation:

1. **Goal information: Basic account info**
![](https://i.imgur.com/Pgln5D1.png)


2. **Find oldest transfer out info**
![](https://i.imgur.com/lXekkQd.png)
    * The captured time will be 8 hours behind the website display. It should be adjusted by the website to match the user's time zone.
3. **Use the oldest transfer address as the next target**
![](https://i.imgur.com/jagMVV6.png)

    - The above steps are performed 4 times in total **(start address -> address 1 -> address 2 -> address 3)** before changing to the next address in the input.txt
    
    - If you encounter an untransferred address on the way, stop early at that address and directly change to the next address in input.txt. 
    - For example: **start address -> address 1 -> address 2 (premature termination)**, and the address stopped early does not have the transfer information, and does not need to grab the oldest transfer information.

