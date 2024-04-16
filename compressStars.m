function [ I_comp ] = compressStars( I_c )
I_comp = zeros(size(I_c));
while(~isempty(find(I_c,1)))
    coord = find(I_c,1);
    I_c1 = zeros(size(I_c));
    checked = zeros(size(I_c));
    I_c1(coord) = I_c(coord);
    [I_c1,checked] = recAdd(I_c,I_c1,coord,checked);
    tot1 = sum(sum(I_c1));
    v = find(I_c1);
    a = length(v);
    compcoord = v(round(a/2));
    val = tot1/a;
    [~,maxC] = max(val,[],3);
    if(maxC~=2)
        I_comp(compcoord) = val;
    end
    I_c = I_c - I_c1;
end
end

