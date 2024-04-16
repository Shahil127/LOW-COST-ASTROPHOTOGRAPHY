function [v] = getSurrounding(I,coord)
        I1 = size(I,1);
        I2 = size(I,2);
        b = ceil(coord./I1);
        a = mod(coord,I1);
        if(a==0)
            a = I2;
        end
        v = [];
        if((a-1)>0 && (b-1)>0)
            v = [v;a-1,b-1,I(a-1,b-1)];
        end
        if((b-1)>0)
            v = [v;a,b-1,I(a,b-1)];
        end 
        if((a+1)<=I1 && (b-1)>0)
            v = [v;a+1,b-1,I(a+1,b-1)];
        end
        if((a-1)>0)
            v = [v;a-1,b,I(a-1,b)];
        end
        if((a+1)<=I1)
            v = [v;a+1,b,I(a+1,b)];
        end
        if((a-1)>0 && (b+1)<=I2)
            v = [v;a-1,b+1,I(a-1,b+1)];
        end
        if((b+1)<=I2)
            v = [v;a,b+1,I(a,b+1)];
        end
        if((a+1)<=I1 && (b+1)<=I2)
            v = [v;a+1,b+1,I(a+1,b+1)];
        end
end
