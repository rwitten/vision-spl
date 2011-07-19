function [avg_pos_dist, avg_pair_dist] = get_avg_psi_dists(psi_pos_file, psi_neg_file)
  pos_psis = dlmread(psi_pos_file);
  neg_psis = dlmread(psi_neg_file);
  pos_psis = pos_psis(:,2:end);
  neg_psis = neg_psis(:,2:end);
avg_pos_dist = zeros(1,5);
avg_pair_dist = zeros(1,5);
for k = 1:5,
	  tot_pos_dist = sum(pdist(pos_psis(:,((k-1)*1000+1):(k*1000))));
tot_neg_dist = sum(pdist(neg_psis(:,((k-1)*1000+1):(k*1000))));
tot_pos_neg_dist = sum(pdist([pos_psis(:,((k-1)*1000+1):(k*1000)) ; neg_psis(:,((k-1)*1000+1):(k*1000))]));
  tot_pair_dist = tot_pos_neg_dist - tot_pos_dist - tot_neg_dist; %if we had Matlab R2010 or later, we could just use pdist2(pos_psis, neg_psis) to get this
    avg_pos_dist(k) = tot_pos_dist / nchoosek(size(pos_psis,1),2);
    avg_pair_dist(k) = tot_pair_dist / (size(pos_psis,1) * size(neg_psis,1));
end
