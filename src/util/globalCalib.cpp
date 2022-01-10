/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/



#include "util/globalCalib.h"
#include "stdio.h"
#include <iostream>

namespace dso
{
	// 2021.12.29
	int wG[PYR_LEVELS], hG[PYR_LEVELS];

	float fxG[CAM_USE][PYR_LEVELS], fyG[CAM_USE][PYR_LEVELS],
		  cxG[CAM_USE][PYR_LEVELS], cyG[CAM_USE][PYR_LEVELS];

	float fxiG[CAM_USE][PYR_LEVELS], fyiG[CAM_USE][PYR_LEVELS],
		  cxiG[CAM_USE][PYR_LEVELS], cyiG[CAM_USE][PYR_LEVELS];

	Eigen::Matrix3f KG[CAM_USE][PYR_LEVELS], KiG[CAM_USE][PYR_LEVELS];

	std::vector<SE3> RT_lb_c(CAM_USE);
	std::vector<SE3> T_c0_c(CAM_USE);

	bool* maskG[CAM_USE][PYR_LEVELS];

	// 2022.1.6
	float* Var_mask[CAM_USE];//
	
	float wM3G;
	float hM3G;

	void setGlobalCalib(int w, int h,const Eigen::Matrix3f &K, int cam_idx, bool* mask)
	{
		// 2021.11.24
		maskG[cam_idx][0] = new bool[w*h];
		for(int i=0;i<w*h;i++){
			maskG[cam_idx][0][i] = mask[i];
		}
		// 2021.11.24
		// 2022.1.6
		Var_mask[cam_idx] = new float[w*h];//

		memset(Var_mask[cam_idx],0.0,sizeof(float)*w*h);

		int wlvl=w;
		int hlvl=h;
		pyrLevelsUsed=1;
		while(wlvl%2==0 && hlvl%2==0 && wlvl*hlvl > 5000 && pyrLevelsUsed < PYR_LEVELS)
		{
			wlvl /=2;
			hlvl /=2;
			pyrLevelsUsed++;
		}
		printf("using pyramid levels 0 to %d. coarsest resolution: %d x %d!\n",
				pyrLevelsUsed-1, wlvl, hlvl);
		if(wlvl>100 && hlvl > 100)
		{
			printf("\n\n===============WARNING!===================\n "
					"using not enough pyramid levels.\n"
					"Consider scaling to a resolution that is a multiple of a power of 2.\n");
		}
		if(pyrLevelsUsed < 3)
		{
			printf("\n\n===============WARNING!===================\n "
					"I need higher resolution.\n"
					"I will probably segfault.\n");
		}

		wM3G = w-3;
		hM3G = h-3;

		wG[0] = w;
		hG[0] = h;

		// 2021.11.8
		KG[cam_idx][0] = K;
		fxG[cam_idx][0] = K(0,0);
		fyG[cam_idx][0] = K(1,1);
		cxG[cam_idx][0] = K(0,2);
		cyG[cam_idx][0] = K(1,2);
		KiG[cam_idx][0] = KG[cam_idx][0].inverse();
		fxiG[cam_idx][0] = KiG[cam_idx][0](0,0);
		fyiG[cam_idx][0] = KiG[cam_idx][0](1,1);
		cxiG[cam_idx][0] = KiG[cam_idx][0](0,2);
		cyiG[cam_idx][0] = KiG[cam_idx][0](1,2);

		for (int level = 1; level < pyrLevelsUsed; ++ level)
		{
			wG[level] = w >> level;
			hG[level] = h >> level;

			fxG[cam_idx][level] = fxG[cam_idx][level-1] * 0.5;
			fyG[cam_idx][level] = fyG[cam_idx][level-1] * 0.5;
			cxG[cam_idx][level] = (cxG[cam_idx][0] + 0.5) / ((int)1<<level) - 0.5;
			cyG[cam_idx][level] = (cyG[cam_idx][0] + 0.5) / ((int)1<<level) - 0.5;

			KG[cam_idx][level]  << fxG[cam_idx][level], 0.0, cxG[cam_idx][level], 0.0, fyG[cam_idx][level], cyG[cam_idx][level], 0.0, 0.0, 1.0;	// synthetic
			KiG[cam_idx][level] = KG[cam_idx][level].inverse();

			fxiG[cam_idx][level] = KiG[cam_idx][level](0,0);
			fyiG[cam_idx][level] = KiG[cam_idx][level](1,1);
			cxiG[cam_idx][level] = KiG[cam_idx][level](0,2);
			cyiG[cam_idx][level] = KiG[cam_idx][level](1,2);
			// 2021.11.24
			// 4合1, 生成金字塔
			maskG[cam_idx][level] = new bool[wG[level]*hG[level]];
			bool* mask_ = maskG[cam_idx][level];
			bool* mask_l = maskG[cam_idx][level-1];
			int wlm1 = wG[level-1]; // 列数
			for(int y=0;y<hG[level];y++)
				for(int x=0;x<wG[level];x++)
				{
					int cnt = 4;
					if(!mask_l[2*x + 2*y*wlm1]){
						cnt --;
					}
					if(!mask_l[2*x+1 + 2*y*wlm1]){
						cnt --;
					}
					if(!mask_l[2*x + 2*y*wlm1+wlm1]){
						cnt --;
					}
					if(!mask_l[2*x+1 + 2*y*wlm1+wlm1]){
						cnt --;
					}
					if(cnt < 4){
						mask_[x + y*wG[level]] = false ;
					}
					else{
						mask_[x + y*wG[level]] = true ;
					}
				}
			// 2021.11.24
		}
		// 2021.11.8
	}
	// 2021.11.14
	Eigen::Matrix4d xyzrpy2T(float ssc[6]){

		float sr = sin(M_PI/180.0 * ssc[3]);
		float cr = cos(M_PI/180.0 * ssc[3]);

		float sp = sin(M_PI/180.0 * ssc[4]);
		float cp = cos(M_PI/180.0 * ssc[4]);

		float sh = sin(M_PI/180.0 * ssc[5]);
		float ch = cos(M_PI/180.0 * ssc[5]);

		Eigen::Vector3d t;
		t << ssc[0], ssc[1], ssc[2];
		Eigen::Matrix3d R;
		R << ch*cp, -sh*cr+ch*sp*sr, sh*sr+ch*sp*cr, sh*cp, ch*cr+sh*sp*sr, -ch*sr+sh*sp*cr, -sp, cp*sr, cp*cr;

		Eigen::Matrix4d T;
		T.setIdentity();

		//2021.11.12
		T.block<3,3>(0,0) = R;
		T.topRightCorner<3, 1>() = t;
		//2021.11.12

		return T;
	}

	void setGlobalExtrin(float ssc[6], int cam_idx){
		Eigen::Matrix4d T = xyzrpy2T(ssc);
		RT_lb_c[cam_idx] = SE3(T.block<3,3>(0,0),T.topRightCorner<3, 1>());
		if(cam_idx==cam_num-1)
			calGlobalExtrin();
	}
	void calGlobalExtrin(){
		for(int i=0;i<cam_num;i++){
			T_c0_c[i] = RT_lb_c[0].inverse()*RT_lb_c[i];
		}
	}
	//
}
