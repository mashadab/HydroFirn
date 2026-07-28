load Achilig_upGPR_data.mat
%﻿May 24 through September 18, 2016 (Samira's data)
%datetime(PercolationDepths201620201{:,1},'ConvertFrom','datenum','Format','dd-MMM-yyyy HH:mm:ss.SSS')
plot(PercolationDepths201620201{851:4200,4}, abs(PercolationDepths201620201{1,2})-abs(PercolationDepths201620201{851:4200,2}),'k-')
datetick('x', 'dd-mmm-yyyy')
ylim([0,2.5]);
set(gca, 'YDir','reverse')