
function mixImages(fixedImg, resultImg, sliceNum, stripWidth, saveFileName)

mixedImg = fixedImg(:,:,sliceNum);

for idx=stripWidth:(2:stripWidth):256
    if( (idx + stripWidth) <= 256)
        mixedImg(:,(idx:(idx+stripWidth))) = resultImg(:,(idx:(idx+stripWidth)), sliceNum);
    else
        mixedImg(:,(idx:end)) = resultImg(:,(idx:end), sliceNum);
    end
end

fhandle = figure(1);
image(mixedImg);
colormap(gray(256));
saveas(fhandle, saveFileName);
