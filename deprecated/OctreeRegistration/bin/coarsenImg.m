
function coarsenImg(fnamePrefix)

imgInfo = analyze75info(fnamePrefix);
imgF = double(analyze75read(imgInfo));

[Ny, Nx, Nz] = size(imgF);

hx = double(imgInfo.PixelDimensions(1));
hy = double(imgInfo.PixelDimensions(2));
hz = double(imgInfo.PixelDimensions(3));

Nxc = Nx/2.0;
Nyc = Ny/2.0;
Nzc = Nz/2.0;

imgC = zeros(Nyc, Nxc, Nzc);

for k = 1:Nzc
    for i = 1:Nxc
        for j = 1:Nyc
            imgC(j, i, k) = 0.125*( imgF(((2*j) - 1), ((2*i) - 1), ((2*k) - 1)) + ...
            imgF(((2*j) - 1), (2*i), ((2*k) - 1)) + ...
            imgF((2*j), ((2*i) - 1), ((2*k) - 1)) + ...
            imgF((2*j), (2*i), ((2*k) - 1)) + ...
            imgF(((2*j) - 1), ((2*i) - 1), (2*k)) + ...
            imgF(((2*j) - 1), (2*i), (2*k)) + ...
            imgF((2*j), ((2*i) - 1), (2*k)) + ...
            imgF((2*j), (2*i), (2*k)) );
        end
    end
end

fnameNewPrefix = [fnamePrefix,'_Coarse'];

saveImgData(imgC, fnameNewPrefix, (2.0*hx), (2.0*hy), (2.0*hz));

display(['New image stored in: ',fnameNewPrefix])





