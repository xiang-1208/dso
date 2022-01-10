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


/*
 * KFBuffer.cpp
 *
 *  Created on: Jan 7, 2014
 *      Author: engelj
 */

#include "FullSystem/FullSystem.h"
 
#include "stdio.h"
#include "util/globalFuncs.h"
#include <Eigen/LU>
#include <algorithm>
#include "IOWrapper/ImageDisplay.h"
#include "util/globalCalib.h"

#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include "FullSystem/ResidualProjections.h"
#include "FullSystem/ImmaturePoint.h"

#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include "IOWrapper/Output3DWrapper.h"

#include "FullSystem/CoarseTracker.h"

namespace dso
{


// 
void FullSystem::flagFramesForMarginalization(FrameHessian* newFH)
{
	//xtl: 1 < 7 不用考虑这个条件 这段是为了只保留setting_maxFrames个关键帧
	if(setting_minFrameAge > setting_maxFrames)
	{
		for(int i=setting_maxFrames;i<(int)frameHessians.size();i++)
		{
			FrameHessian* fh = frameHessians[i-setting_maxFrames];
			// 2021.11.15
			for(int i=0;i<cam_num;i++)
				fh->frame[i]->flaggedForMarginalization = true;
			// 2021.11.15
		}
		return;
	}


	int flagged = 0;
	// marginalize all frames that have not enough points.
	for(int i=0;i<(int)frameHessians.size();i++)
	{
		int in = 0;
		int out = 0;
		FrameHessian* fh = frameHessians[i];
		for(int idx=0;idx<cam_num;idx++){
			in += fh->frame[idx]->pointHessians.size() + fh->frame[idx]->immaturePoints.size(); // 还在的点
			out += fh->frame[idx]->pointHessiansMarginalized.size() + fh->frame[idx]->pointHessiansOut.size(); // 边缘化和丢掉的点
		}
		Vec2 refToFh=AffLight::fromToVecExposure(frameHessians.back()->ab_exposure, fh->ab_exposure,
				frameHessians.back()->aff_g2l(), fh->aff_g2l());

		// 论文中提到的，若内点数量占比过小/光度变化大于阈值，并且现在还有足够的帧，就标记该帧需要边缘化
		if( (in < setting_minPointsRemaining *(in+out) || fabs(logf((float)refToFh[0])) > setting_maxLogAffFacInWindow)
				&& ((int)frameHessians.size())-flagged > setting_minFrames)
		{
			// 2021.11.15
			for(int idx=0;idx<cam_num;idx++)
				fh->frame[idx]->flaggedForMarginalization = true;
			flagged++;
		}
		else
		{
//			printf("May Keep frame %d, as %'d/%'d points remaining (%'d %'d %'d %'d). VisInLast %'d / %'d. traces %d, activated %d!\n",
//					fh->frameID, in, in+out,
//					(int)fh->pointHessians.size(), (int)fh->immaturePoints.size(),
//					(int)fh->pointHessiansMarginalized.size(), (int)fh->pointHessiansOut.size(),
//					visInLast, outInLast,
//					fh->statistics_tracesCreatedForThisFrame, fh->statistics_pointsActivatedForThisFrame);
		}
	}

	// 若减去标记了边缘化的帧，frameHessians的总帧数还大于窗口中允许的最大帧数
	// marginalize one.
	if((int)frameHessians.size()-flagged >= setting_maxFrames)
	{
		double smallestScore = 1;
		FrameHessian* toMarginalize=0;
		FrameHessian* latest = frameHessians.back();


		for(FrameHessian* fh : frameHessians)
		{
			if(fh->frameID > latest->frameID-setting_minFrameAge || fh->frameID == 0) continue;
			//if(fh==frameHessians.front() == 0) continue;

			double distScore = 0;
			for(FrameFramePrecalc &ffh : fh->targetPrecalc)
			{
				if(ffh.target->frameID > latest->frameID-setting_minFrameAge+1 || ffh.target == ffh.host) continue;
				distScore += 1/(1e-5+ffh.distanceLL);

			}
			distScore *= -sqrtf(fh->targetPrecalc.back().distanceLL);


			if(distScore < smallestScore)
			{
				smallestScore = distScore;
				toMarginalize = fh;
			}
		}

//		printf("MARGINALIZE frame %d, as it is the closest (score %.2f)!\n",
//				toMarginalize->frameID, smallestScore);
		// 2021.11.15
		for(int i=0;i<cam_num;i++){
			toMarginalize->frame[i]->flaggedForMarginalization = true;
		}
		flagged++;
	}

//	printf("FRAMES LEFT: ");
//	for(FrameHessian* fh : frameHessians)
//		printf("%d ", fh->frameID);
//	printf("\n");
}



// XTL：边缘化一帧，先边缘化该帧的efFrame：对ef的H、b做了操作;然后遍历所有残差项，若其目标帧为该帧，就调整其主点的lastResiduals，并且在主点中丢掉该残差
// XTL：在该帧的shell里面记录下边缘化时滑窗里面的最后一帧，且记录下w2c_leftEps。将该帧的所有efFrame置零，然后删掉frameHessians中的该帧，重新给滑窗中的帧编号，计算adHost,adTarget
void FullSystem::marginalizeFrame(FrameHessian* frame)
{
	// marginalize or remove all this frames points.

	// 2021.11.16
	for(int i=0;i<cam_num;i++)
		assert((int)frame->frame[i]->pointHessians.size()==0);

	ef->marginalizeFrame(frame->efFrame);

	for(FrameHessian* fh : frameHessians)
	{
		if(fh==frame) continue;
		for(int idx=0;idx<cam_num;idx++)
			for(PointHessian* ph : fh->frame[idx]->pointHessians)
			{
				for(unsigned int i=0;i<ph->residuals.size();i++)
				{
					PointFrameResidual* r = ph->residuals[i];
					if(r->target == frame->frame[idx])
					{
						if(ph->lastResiduals[0].first == r)
							ph->lastResiduals[0].first=0;
						else if(ph->lastResiduals[1].first == r)
							ph->lastResiduals[1].first=0;


						if(r->host->frameID < r->target->frameID)
							statistics_numForceDroppedResFwd++;
						else
							statistics_numForceDroppedResBwd++;

						ef->dropResidual(r->efResidual);
						deleteOut<PointFrameResidual>(ph->residuals,i);
						break;
					}
				}
			}
	}



    {
        std::vector<FrameHessian*> v;
        v.push_back(frame);
        for(IOWrap::Output3DWrapper* ow : outputWrapper)
            ow->publishKeyframes(v, true, &Hcalib);
    }

	frame->shell->marginalizedAt = frameHessians.back()->shell->id;
	frame->shell->movedByOpt = frame->w2c_leftEps().norm();

	// 2021.11.26
	for(int i=1;i<cam_num;i++){
		frame->frame[i]->efFrame=0;
		delete frame->frame[i];
	}
	frame->efFrame=0;

	deleteOutOrder<FrameHessian>(frameHessians, frame);
	// 2021.11.23
	for(unsigned int i=0;i<frameHessians.size();i++){
		for(int _idx=0;_idx<cam_num;_idx++){
			frameHessians[i]->frame[_idx]->idx = i;
		}
	}
	// 2021.11.23


	setPrecalcValues();
	ef->setAdjointsF(&Hcalib);
}




}
