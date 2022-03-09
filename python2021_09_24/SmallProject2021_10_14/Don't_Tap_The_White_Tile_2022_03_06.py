'''
Author: xudawu
Date: 2022-03-06 20:33:03
LastEditors: xudawu
LastEditTime: 2022-03-08 17:20:37
'''
import pyautogui
import keyboard
while True:
    if keyboard.is_pressed('Q'):
        break
    x=80
    y=630
    if pyautogui.pixelMatchesColor(x, y,(32,32,32)) == True:
        pyautogui.click(x,y,button='left')
    x=230
    if pyautogui.pixelMatchesColor(x, y, (32, 32, 32)) == True:
        pyautogui.click(x, y, button='left')
    x=390
    if pyautogui.pixelMatchesColor(x, y, (32, 32, 32)) == True:
        pyautogui.click(x, y, button='left')
    x=580
    if pyautogui.pixelMatchesColor(x, y, (32, 32, 32)) == True:
        pyautogui.click(x, y, button='left')
