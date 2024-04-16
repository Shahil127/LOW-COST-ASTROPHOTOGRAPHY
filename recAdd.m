function [ I_c,checked] = recAdd( I,I_c,coord,checked )
v = getSurrounding(I,coord);
vals = v(:,3);
checked(coord) = 1;
I1 = size(I,1);
I2 = size(I,2);
if(~isempty(find(vals,1)))
    for a = 1:size(v,1)
        x1 = v(a,1);
        y1 = v(a,2);
        I_c(x1,y1) = I(x1,y1);
        coord1 = (y1-1).*I1 + x1;
        if(checked(x1,y1)==0)
            [I_c,checked] = recAdd(I,I_c,coord1,checked);
        end
    end
end
end

