
function viewMismatch(imgFixed, imgMoving, imgResult)
 
colorScale = 256;

[N, dummy, dummy] = size(imgFixed);

initMismatch = abs(imgFixed - imgMoving);
finalMismatch = abs(imgFixed - imgResult);

scaleFac = max(max(max(initMismatch)));

initMismatch = (colorScale-1)*(initMismatch/scaleFac);
finalMismatch = (colorScale-1)*(finalMismatch/scaleFac);

for i = 1:N            
    scrsz = get(0,'ScreenSize');
    fhandle = figure(1);
    set(fhandle, 'Position',[1 scrsz(4)/2 scrsz(3)/2 scrsz(4)/2])
    
    axis off;
    
    subplot(1, 2, 1);
    image(initMismatch(:,:,i));
    title('Initial Mismatch')
    colormap(gray(colorScale));
    
    axis off;
    
    subplot(1, 2, 2);
    image(finalMismatch(:,:,i));
    title('Final Mismatch')
    colormap(gray(colorScale));
        
    fname = ['slice_', int2str(i), '.jpg'];
    saveas(fhandle, fname);
end

