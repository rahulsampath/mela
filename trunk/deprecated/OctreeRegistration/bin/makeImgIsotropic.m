
function makeImgIsotropic(fnamePrefix)

imgInfo = analyze75info(fnamePrefix);
imgVals = double(analyze75read(imgInfo));

[Ny, Nx, Nz] = size(imgVals)

hx = double(imgInfo.PixelDimensions(1))
hy = double(imgInfo.PixelDimensions(2))
hz = double(imgInfo.PixelDimensions(3))

xVec = hx*(0:(Nx - 1));
yVec = hy*(0:(Ny - 1));
zVec = hz*(0:(Nz - 1));

[xx, yy, zz] = meshgrid(xVec, yVec, zVec);

h = min([hx; hy; hz])

Nxi = floor(1.0 + (hx/h)*(Nx - 1))
Nyi = floor(1.0 + (hy/h)*(Ny - 1))
Nzi = floor(1.0 + (hz/h)*(Nz - 1))

xi = h*(0:(Nxi - 1));
yi = h*(0:(Nyi - 1));
zi = h*(0:(Nzi - 1));

[xxi, yyi, zzi] = meshgrid(xi, yi, zi);

isotropicImg = interp3(xx, yy, zz, imgVals, xxi, yyi, zzi);

fnameNewPrefix = [fnamePrefix,'_Isotropic'];

saveImgData(isotropicImg, fnameNewPrefix, h, h, h);

display(['New image stored in: ',fnameNewPrefix])





