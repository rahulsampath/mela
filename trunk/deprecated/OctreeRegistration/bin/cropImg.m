
function cropImg(fnamePrefix, Ne)

imgInfo = analyze75info(fnamePrefix);
imgVals = double(analyze75read(imgInfo));

[Ny, Nx, Nz] = size(imgVals);

hx = double(imgInfo.PixelDimensions(1));
hy = double(imgInfo.PixelDimensions(2));
hz = double(imgInfo.PixelDimensions(3));

xPad = (Nx - Ne)/2;
yPad = (Ny - Ne)/2;

croppedImg = imgVals(((yPad + 1):(Ny - yPad)), ((xPad + 1):(Nx - xPad)), :);

fnameNewPrefix = [fnamePrefix,'_Cropped'];

saveImgData(croppedImg, fnameNewPrefix, hx, hy, hz);

display(['New image stored in: ',fnameNewPrefix])





