
function zeroPadImg(fnamePrefix, Ne)

imgInfo = analyze75info(fnamePrefix);
imgVals = double(analyze75read(imgInfo));

[Ny, Nx, Nz] = size(imgVals);

hx = double(imgInfo.PixelDimensions(1));
hy = double(imgInfo.PixelDimensions(2));
hz = double(imgInfo.PixelDimensions(3));

xPadl = floor((Ne - Nx)/2)
yPadl = floor((Ne - Ny)/2)
zPadl = floor((Ne - Nz)/2)

xPadr = Ne - Nx - xPadl
yPadr = Ne - Ny - yPadl
zPadr = Ne - Nz - zPadl

newImg = zeros(Ne, Ne, Ne);

newImg(((yPadl + 1):(Ne - yPadr)), ((xPadl + 1):(Ne - xPadr)), ((zPadl + 1):(Ne - zPadr))) = imgVals;

fnameNewPrefix = [fnamePrefix,'_New'];

saveImgData(newImg, fnameNewPrefix, hx, hy, hz);

display(['New image stored in: ',fnameNewPrefix])





