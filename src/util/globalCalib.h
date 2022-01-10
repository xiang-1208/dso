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



#pragma once
#include "util/settings.h"
#include "util/NumType.h"

namespace dso
{
	// 2021.11.8
	extern int wG[PYR_LEVELS], hG[PYR_LEVELS];
	extern float fxG[CAM_USE][PYR_LEVELS], fyG[CAM_USE][PYR_LEVELS],
		  cxG[CAM_USE][PYR_LEVELS], cyG[CAM_USE][PYR_LEVELS];

	extern float fxiG[CAM_USE][PYR_LEVELS], fyiG[CAM_USE][PYR_LEVELS],
		  cxiG[CAM_USE][PYR_LEVELS], cyiG[CAM_USE][PYR_LEVELS];

	extern Eigen::Matrix3f KG[CAM_USE][PYR_LEVELS],KiG[CAM_USE][PYR_LEVELS];

	extern std::vector<SE3> T_c0_c;
	extern std::vector<SE3> RT_lb_c;
	//

	// 2021.11.24
	extern bool* maskG[CAM_USE][PYR_LEVELS];
	//
	// 2022.1.6
	extern float* Var_mask[CAM_USE];//

	extern float wM3G;
	extern float hM3G;

	void setGlobalCalib(int w, int h, const Eigen::Matrix3f &K, int cam_idx, bool* mask);

	Eigen::Matrix4d xyzrpy2T(float ssc[6]);
	void setGlobalExtrin(float ssc[6], int cam_idx);
	void calGlobalExtrin();
}
