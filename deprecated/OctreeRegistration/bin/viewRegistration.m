
function viewRegistration(imgFixed, imgMoving, imgResult)
 
[N, dummy, dummy] = size(imgFixed);

for i = 1:N        
    
    scrsz = get(0,'ScreenSize');
    fhandle = figure(1);
    set(fhandle, 'Position',[1 scrsz(4)/2 scrsz(3)/2 scrsz(4)/2])
    
    axis off;
    
    subplot(2, 2, 1);
    image(imgFixed(:,:,i));
    title('Fixed')
    colormap(gray(256));
    
    axis off;
    
    subplot(2, 2, 3);
    image(imgMoving(:,:,i));
    title('Moving')
    colormap(gray(256));
    
    axis off;
    
    subplot(2, 2, 4);
    image(imgResult(:,:,i));
    title('Registered')
    colormap(gray(256));
    
    axis off;
    
    fname = ['slice_', int2str(i), '.jpg'];
    saveas(fhandle, fname);
end

