## front, rightf, leftf, leftrear, rear, rightrear, 

<!-- Roof -->
<!-- alloywheel -->
<!-- antenna -->
<!-- bonnet -->
<!-- broken -->
<!-- bumperdent
bumpertear
bumpertorn -->
<!-- clipsbroken -->
<!-- d2 -->
<!-- dirt -->
<!-- doorhandle -->
<!-- frontbumper
frontbumpergrille -->
<!-- frontws -->
<!-- fuelcap -->
<!-- indicator -->
<!-- leftapillar
leftcpillar
leftfender -->
<!-- leftfoglamp -->
<!-- leftfrontdoor
leftfrontdoorcladding
leftfrontdoorglass -->
<!-- leftheadlamp -->
<!-- leftorvm -->
<!-- leftqpanel -->
<!-- leftreardoor
leftreardoorcladding
leftreardoorglass -->
<!-- leftroofside
leftrunningboard -->
<!-- lefttaillamp -->
<!-- leftwa -->
<!-- licenseplate -->
<!-- logo -->
<!-- lowerbumpergrille -->
<!-- namebadge -->
<!-- partial_bonnet  -->
<!-- partial_frontbumper
partial_frontws -->
<!-- partial_leftfender -->
<!-- partial_leftfrontdoor
partial_leftreardoor -->
<!-- partial_rearbumper
partial_rearws
partial_rightfender
partial_rightfrontdoor
partial_rightqpanel
partial_rightreardoor
partial_tailgate -->
<!-- rearbumper
rearws -->
<!-- rightapillar -->
<!-- rightbpillar -->
<!-- rightcpillar -->
<!-- rightfender -->
<!-- rightfoglamp -->
<!-- rightfrontdoor
rightfrontdoorcladding
rightfrontdoorglass -->
<!-- rightheadlamp
rightorvm
rightqpanel -->
<!-- rightreardoor
rightreardoorcladding
rightreardoorglass
rightrearventglass -->
<!-- rightrunningboard
righttaillamp
rightwa -->
<!-- scratch
sensor -->
<!-- tailgate
towbarcover -->
<!-- tyre -->
<!-- wheelcap -->
<!-- wheelrim -->
<!-- wiper -->


## front 
Roof
bumper
frontbumper
frontbumpergrille
frontws
bumperdent
bumpertear
bumpertorn
partial_frontbumper
partial_frontws

## leftfront

leftapillar
leftfender
leftfrontdoor
leftfrontdoorcladding
leftfrontdoorglass
leftorvm
leftroofside


## leftrear

leftqpanel
leftcpillar
leftreardoor
leftreardoorcladding
leftreardoorglass

## rightfront

rightorvm
rightapillar
rightfender
rightfrontdoor
rightfrontdoorcladding
rightfrontdoorglass

## rightrear
rightqpanel
rightreardoor
rightreardoorcladding
rightreardoorglass
rightrearventglass
righttaillamp

## rear
antenna
rearbumper
rearws
tailgate
towbarcover


#cycle,human, tyre

front - 1278
leftfront - 555
leftrear - 261 
rear - 762
rightfront - 495
rightrear - 405
none - 574

1.get all the identities from json file and seperate it to front, rear, leftrear, rightrear, leftfront, rightfront
2.created seperate directory files
3. created a "none" dataset with images of human,cycle, garage

Epoch 20/20  efficientnet_lite0  14mb
Val Loss : 0.9684
Acc      : 78.04%
Precision: 0.757
Recall   : 0.757


python3 -m venv myenv
source myenv/bin/activate
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt

onnx2tf -i efficientnet_best.onnx
python 6.int8_conversion.py



