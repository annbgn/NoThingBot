- when game starts and it's window opens we've gotta check if there is no opened instance for this game already
- exit game in the end (wrap process?)
- make PATH not hardcoded but think of heuristic algorithm which is very likely to find the game on any person PC in steam/steamapps/common/NO THING folder
- requirements (Pillow,  pywinauto, Numpy)
- more flexible screen sizes
- add log file
- auto compute good canny threshold values https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/ 
  - problem: low contrast screens tend to have no edges after applying canny. solution idea: make contrast higher

