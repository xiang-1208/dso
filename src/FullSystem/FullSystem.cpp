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
#include "FullSystem/PixelSelector.h"
#include "FullSystem/PixelSelector2.h"
#include "FullSystem/ResidualProjections.h"
#include "FullSystem/ImmaturePoint.h"

#include "FullSystem/CoarseTracker.h"
#include "FullSystem/CoarseInitializer.h"

#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include "IOWrapper/Output3DWrapper.h"

#include "util/ImageAndExposure.h"

#include <cmath>

// 2021.10.28
#include "opencv2/highgui/highgui.hpp"
// 2021.10.28

namespace dso
{
int FrameHessian::instanceCounter=0;
int PointHessian::instanceCounter=0;
int CalibHessian::instanceCounter=0;



FullSystem::FullSystem()
{

	int retstat =0;
	if(setting_logStuff)
	{

		retstat += system("rm -rf logs");
		retstat += system("mkdir logs");

		retstat += system("rm -rf mats");
		retstat += system("mkdir mats");

		calibLog = new std::ofstream();
		calibLog->open("logs/calibLog.txt", std::ios::trunc | std::ios::out);
		calibLog->precision(12);

		numsLog = new std::ofstream();
		numsLog->open("logs/numsLog.txt", std::ios::trunc | std::ios::out);
		numsLog->precision(10);

		coarseTrackingLog = new std::ofstream();
		coarseTrackingLog->open("logs/coarseTrackingLog.txt", std::ios::trunc | std::ios::out);
		coarseTrackingLog->precision(10);

		eigenAllLog = new std::ofstream();
		eigenAllLog->open("logs/eigenAllLog.txt", std::ios::trunc | std::ios::out);
		eigenAllLog->precision(10);

		eigenPLog = new std::ofstream();
		eigenPLog->open("logs/eigenPLog.txt", std::ios::trunc | std::ios::out);
		eigenPLog->precision(10);

		eigenALog = new std::ofstream();
		eigenALog->open("logs/eigenALog.txt", std::ios::trunc | std::ios::out);
		eigenALog->precision(10);

		DiagonalLog = new std::ofstream();
		DiagonalLog->open("logs/diagonal.txt", std::ios::trunc | std::ios::out);
		DiagonalLog->precision(10);

		variancesLog = new std::ofstream();
		variancesLog->open("logs/variancesLog.txt", std::ios::trunc | std::ios::out);
		variancesLog->precision(10);


		nullspacesLog = new std::ofstream();
		nullspacesLog->open("logs/nullspacesLog.txt", std::ios::trunc | std::ios::out);
		nullspacesLog->precision(10);
	}
	else
	{
		nullspacesLog=0;
		variancesLog=0;
		DiagonalLog=0;
		eigenALog=0;
		eigenPLog=0;
		eigenAllLog=0;
		numsLog=0;
		calibLog=0;
	}

	assert(retstat!=293847);


	// 2021.11.14
	//for(int i=0;i<cam_num;i++){
		selectionMap = new float[wG[0]*hG[0]];
	//}
	// 2021.11.14

	coarseDistanceMap = new CoarseDistanceMap(wG[0], hG[0]);
	coarseTracker = new CoarseTracker(wG[0], hG[0]);
	coarseTracker_forNewKF = new CoarseTracker(wG[0], hG[0]);
	coarseInitializer = new CoarseInitializer(wG[0], hG[0]);
	pixelSelector = new PixelSelector(wG[0], hG[0]);

	statistics_lastNumOptIts=0;
	statistics_numDroppedPoints=0;
	statistics_numActivatedPoints=0;
	statistics_numCreatedPoints=0;
	statistics_numForceDroppedResBwd = 0;
	statistics_numForceDroppedResFwd = 0;
	statistics_numMargResFwd = 0;
	statistics_numMargResBwd = 0;

	lastCoarseRMSE.setConstant(100);

	currentMinActDist=2;
	initialized=false;


	ef = new EnergyFunctional();
	ef->red = &this->treadReduce;

	isLost=false;
	initFailed=false;


	needNewKFAfter = -1;

	linearizeOperation=true;
	runMapping=true;
	mappingThread = boost::thread(&FullSystem::mappingLoop, this);
	lastRefStopID=0;



	minIdJetVisDebug = -1;
	maxIdJetVisDebug = -1;
	minIdJetVisTracker = -1;
	maxIdJetVisTracker = -1;
}

FullSystem::~FullSystem()
{
	blockUntilMappingIsFinished();

	if(setting_logStuff)
	{
		calibLog->close(); delete calibLog;
		numsLog->close(); delete numsLog;
		coarseTrackingLog->close(); delete coarseTrackingLog;
		//errorsLog->close(); delete errorsLog;
		eigenAllLog->close(); delete eigenAllLog;
		eigenPLog->close(); delete eigenPLog;
		eigenALog->close(); delete eigenALog;
		DiagonalLog->close(); delete DiagonalLog;
		variancesLog->close(); delete variancesLog;
		nullspacesLog->close(); delete nullspacesLog;
	}

	// for(int i=0;i<cam_num;i++){
		delete[] selectionMap;
	// }

	for(FrameShell* s : allFrameHistory)
		delete s;
	for(FrameHessian* fh : unmappedTrackedFrames){
		// 2021.12.01
		for(int i=1;i<cam_num;i++)
			delete fh->frame[i];
		// 2021.12.01
		delete fh;
	}

	delete coarseDistanceMap;
	delete coarseTracker;
	delete coarseTracker_forNewKF;
	delete coarseInitializer;
	delete pixelSelector;
	delete ef;
}

void FullSystem::setOriginalCalib(const VecXf &originalCalib, int originalW, int originalH)
{

}

void FullSystem::setGammaFunction(float* BInv)
{
	if(BInv==0) return;

	// copy BInv.
	memcpy(Hcalib.Binv, BInv, sizeof(float)*256);


	// invert.
	for(int i=1;i<255;i++)
	{
		// find val, such that Binv[val] = i.
		// I dont care about speed for this, so do it the stupid way.

		for(int s=1;s<255;s++)
		{
			if(BInv[s] <= i && BInv[s+1] >= i)
			{
				Hcalib.B[i] = s+(i - BInv[s]) / (BInv[s+1]-BInv[s]);
				break;
			}
		}
	}
	Hcalib.B[0] = 0;
	Hcalib.B[255] = 255;
}



void FullSystem::printResult(std::string file)
{
	boost::unique_lock<boost::mutex> lock(trackMutex);
	boost::unique_lock<boost::mutex> crlock(shellPoseMutex);

	std::ofstream myfile;
	myfile.open (file.c_str());
	myfile << std::setprecision(15);

	for(FrameShell* s : allFrameHistory)
	{
		if(!s->poseValid) continue;

		if(setting_onlyLogKFPoses && s->marginalizedAt == s->id) continue;

		myfile << s->timestamp <<
			" " << s->camToWorld.translation().transpose()<<
			" " << s->camToWorld.so3().unit_quaternion().x()<<
			" " << s->camToWorld.so3().unit_quaternion().y()<<
			" " << s->camToWorld.so3().unit_quaternion().z()<<
			" " << s->camToWorld.so3().unit_quaternion().w() << "\n";
	}
	myfile.close();
}

// XTL：尝试一系列的运动，进行最优初值的选择，更新camToTrackingRef，camToWorld
Vec4 FullSystem::trackNewCoarse(FrameHessian* fh)
{

	assert(allFrameHistory.size() > 0);
	// set pose initialization.

    for(IOWrap::Output3DWrapper* ow : outputWrapper)
        ow->pushLiveFrame(fh);

	// printf("trackNewCoarse-a");

	FrameHessian* lastF = coarseTracker->lastRef;

	AffLight aff_last_2_l = AffLight(0,0);

	std::vector<SE3,Eigen::aligned_allocator<SE3>> lastF_2_fh_tries;
	// XTL：若只有两帧（最开始调用该函数的情况），则认为这两帧之间无运动。
	if(allFrameHistory.size() == 2)
		for(unsigned int i=0;i<lastF_2_fh_tries.size();i++) lastF_2_fh_tries.push_back(SE3());
	else // XTL：将上一帧到大上帧的运动作为这帧到上帧的运动的粗略初值
	{
		FrameShell* slast = allFrameHistory[allFrameHistory.size()-2];
		FrameShell* sprelast = allFrameHistory[allFrameHistory.size()-3];
		SE3 slast_2_sprelast;
		SE3 lastF_2_slast;
		{	// lock on global pose consistency!
			boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
			slast_2_sprelast = sprelast->camToWorld.inverse() * slast->camToWorld;
			lastF_2_slast = slast->camToWorld.inverse() * lastF->shell->camToWorld;
			aff_last_2_l = slast->aff_g2l;
		}
		SE3 fh_2_slast = slast_2_sprelast;// assumed to be the same as fh_2_slast.


		// get last delta-movement.
		lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast);	// assume constant motion.
		lastF_2_fh_tries.push_back(fh_2_slast.inverse() * fh_2_slast.inverse() * lastF_2_slast);	// assume double motion (frame skipped)
		lastF_2_fh_tries.push_back(SE3::exp(fh_2_slast.log()*0.5).inverse() * lastF_2_slast); // assume half motion.
		lastF_2_fh_tries.push_back(lastF_2_slast); // assume zero motion.
		lastF_2_fh_tries.push_back(SE3()); // assume zero motion FROM KF.


		// just try a TON of different initializations (all rotations). In the end,
		// if they don't work they will only be tried on the coarsest level, which is super fast anyway.
		// also, if tracking rails here we loose, so we really, really want to avoid that.
		for(float rotDelta=0.02; rotDelta < 0.05; rotDelta++)
		{
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,0,0), Vec3(0,0,0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,rotDelta,0), Vec3(0,0,0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,0,rotDelta), Vec3(0,0,0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,0,0), Vec3(0,0,0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,-rotDelta,0), Vec3(0,0,0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,0,-rotDelta), Vec3(0,0,0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,0,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,0,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,0,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,0,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
		}

		if(!slast->poseValid || !sprelast->poseValid || !lastF->shell->poseValid)
		{
			lastF_2_fh_tries.clear();
			lastF_2_fh_tries.push_back(SE3());
		}
	}


	Vec3 flowVecs = Vec3(100,100,100);
	SE3 lastF_2_fh = SE3();
	AffLight aff_g2l = AffLight(0,0);


	// as long as maxResForImmediateAccept is not reached, I'll continue through the options.
	// I'll keep track of the so-far best achieved residual for each level in achievedRes.
	// If on a coarse level, tracking is WORSE than achievedRes, we will not continue to save time.

	// printf("trackNewCoarse-b");

	Vec5 achievedRes = Vec5::Constant(NAN);
	bool haveOneGood = false;
	int tryIterations=0;
	for(unsigned int i=0;i<lastF_2_fh_tries.size();i++)
	{
		AffLight aff_g2l_this = aff_last_2_l;
		SE3 lastF_2_fh_this = lastF_2_fh_tries[i];
		bool trackingIsGood = coarseTracker->trackNewestCoarse(
				fh, lastF_2_fh_this, aff_g2l_this,
				pyrLevelsUsed-1,
				achievedRes);	// in each level has to be at least as good as the last try.
		tryIterations++;

		if(i != 0)
		{
			/*
			printf("RE-TRACK ATTEMPT %d with initOption %d and start-lvl %d (ab %f %f): %f %f %f %f %f -> %f %f %f %f %f \n",
					i,
					i, pyrLevelsUsed-1,
					aff_g2l_this.a,aff_g2l_this.b,
					achievedRes[0],
					achievedRes[1],
					achievedRes[2],
					achievedRes[3],
					achievedRes[4],
					coarseTracker->lastResiduals[0],
					coarseTracker->lastResiduals[1],
					coarseTracker->lastResiduals[2],
					coarseTracker->lastResiduals[3],
					coarseTracker->lastResiduals[4]);
			*/
		}


		// do we have a new winner?
		if(trackingIsGood && std::isfinite((float)coarseTracker->lastResiduals[0]) && !(coarseTracker->lastResiduals[0] >=  achievedRes[0]))
		{
			//printf("take over. minRes %f -> %f!\n", achievedRes[0], coarseTracker->lastResiduals[0]);
			flowVecs = coarseTracker->lastFlowIndicators;
			aff_g2l = aff_g2l_this;
			lastF_2_fh = lastF_2_fh_this;
			haveOneGood = true;
		}

		// take over achieved res (always).
		if(haveOneGood)
		{
			for(int i=0;i<5;i++)
			{
				if(!std::isfinite((float)achievedRes[i]) || achievedRes[i] > coarseTracker->lastResiduals[i])	// take over if achievedRes is either bigger or NAN.
					achievedRes[i] = coarseTracker->lastResiduals[i];
			}
		}


        if(haveOneGood &&  achievedRes[0] < lastCoarseRMSE[0]*setting_reTrackThreshold)
            break;

	}

	if(!haveOneGood)
	{
        printf("BIG ERROR! tracking failed entirely. Take predictred pose and hope we may somehow recover.\n");
		flowVecs = Vec3(0,0,0);
		aff_g2l = aff_last_2_l;
		lastF_2_fh = lastF_2_fh_tries[0];
	}

	lastCoarseRMSE = achievedRes;

	// no lock required, as fh is not used anywhere yet.
	fh->shell->camToTrackingRef = lastF_2_fh.inverse();
	fh->shell->trackingRef = lastF->shell;
	fh->shell->aff_g2l = aff_g2l;
	fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;


	if(coarseTracker->firstCoarseRMSE < 0)
		coarseTracker->firstCoarseRMSE = achievedRes[0];

    if(!setting_debugout_runquiet)
        printf("Coarse Tracker tracked ab = %f %f (exp %f). Res %f!\n", aff_g2l.a, aff_g2l.b, fh->ab_exposure, achievedRes[0]);



	if(setting_logStuff)
	{
		(*coarseTrackingLog) << std::setprecision(16)
						<< fh->shell->id << " "
						<< fh->shell->timestamp << " "
						<< fh->ab_exposure << " "
						<< fh->shell->camToWorld.log().transpose() << " "
						<< aff_g2l.a << " "
						<< aff_g2l.b << " "
						<< achievedRes[0] << " "
						<< tryIterations << "\n";
	}


	return Vec4(achievedRes[0], flowVecs[0], flowVecs[1], flowVecs[2]);
}

void FullSystem::traceNewCoarse(FrameHessian* fh)
{
	boost::unique_lock<boost::mutex> lock(mapMutex);

	int trace_total=0, trace_good=0, trace_oob=0, trace_out=0, trace_skip=0, trace_badcondition=0, trace_uninitialized=0;

	Mat33f K = Mat33f::Identity();

	// 2021.11.13
	for(int i=0;i<cam_num;i++)
	{
		K(0,0) = Hcalib.fxl(i);
		K(1,1) = Hcalib.fyl(i);
		K(0,2) = Hcalib.cxl(i);
		K(1,2) = Hcalib.cyl(i);

		for(FrameHessian* host : frameHessians)		// go through all active frames
		{

			// 2022.1.9
			SE3 hostToNew = fh->frame[i]->PRE_worldToCam * host->frame[i]->PRE_camToWorld;
			Mat33f KRKi = K * hostToNew.rotationMatrix().cast<float>() * K.inverse();
			Vec3f Kt = K * hostToNew.translation().cast<float>();

			Vec2f aff = AffLight::fromToVecExposure(host->ab_exposure, fh->ab_exposure, host->aff_g2l(), fh->aff_g2l()).cast<float>();

			for(ImmaturePoint* ph : host->frame[i]->immaturePoints)
			{
				ph->traceOn(fh->frame[i], KRKi, Kt, aff, &Hcalib, false );

				if(ph->lastTraceStatus==ImmaturePointStatus::IPS_GOOD) trace_good++;
				if(ph->lastTraceStatus==ImmaturePointStatus::IPS_BADCONDITION) trace_badcondition++;
				if(ph->lastTraceStatus==ImmaturePointStatus::IPS_OOB) trace_oob++;
				if(ph->lastTraceStatus==ImmaturePointStatus::IPS_OUTLIER) trace_out++;
				if(ph->lastTraceStatus==ImmaturePointStatus::IPS_SKIPPED) trace_skip++;
				if(ph->lastTraceStatus==ImmaturePointStatus::IPS_UNINITIALIZED) trace_uninitialized++;
				trace_total++;
			}
		}
		/*
		printf("CAM %d,ADD: TRACE: %'d points. %'d (%.0f%%) good. %'d (%.0f%%) skip. %'d (%.0f%%) badcond. %'d (%.0f%%) oob. %'d (%.0f%%) out. %'d (%.0f%%) uninit.\n",
				i,
				trace_total,
				trace_good, 100*trace_good/(float)trace_total,
				trace_skip, 100*trace_skip/(float)trace_total,
				trace_badcondition, 100*trace_badcondition/(float)trace_total,
				trace_oob, 100*trace_oob/(float)trace_total,
				trace_out, 100*trace_out/(float)trace_total,
				trace_uninitialized, 100*trace_uninitialized/(float)trace_total);*/
	}
}




void FullSystem::activatePointsMT_Reductor(
		std::vector<PointHessian*>* optimized,
		std::vector<ImmaturePoint*>* toOptimize,
		int min, int max, Vec10* stats, int tid)
{
	ImmaturePointTemporaryResidual* tr = new ImmaturePointTemporaryResidual[frameHessians.size()];
	for(int k=min;k<max;k++)
	{
		(*optimized)[k] = optimizeImmaturePoint((*toOptimize)[k],1,tr);
	}
	delete[] tr;
}



void FullSystem::activatePointsMT()
{
	// 2021.10.31
	setting_desiredPointDensity *= cam_num;
	// 2021.10.31
	if(ef->nPoints < setting_desiredPointDensity*0.66)
		currentMinActDist -= 0.8;
	if(ef->nPoints < setting_desiredPointDensity*0.8)
		currentMinActDist -= 0.5;
	else if(ef->nPoints < setting_desiredPointDensity*0.9)
		currentMinActDist -= 0.2;
	else if(ef->nPoints < setting_desiredPointDensity)
		currentMinActDist -= 0.1;

	if(ef->nPoints > setting_desiredPointDensity*1.5)
		currentMinActDist += 0.8;
	if(ef->nPoints > setting_desiredPointDensity*1.3)
		currentMinActDist += 0.5;
	if(ef->nPoints > setting_desiredPointDensity*1.15)
		currentMinActDist += 0.2;
	if(ef->nPoints > setting_desiredPointDensity)
		currentMinActDist += 0.1;

	if(currentMinActDist < 0) currentMinActDist = 0;
	if(currentMinActDist > 4) currentMinActDist = 4;

    if(!setting_debugout_runquiet)
        printf("SPARSITY:  MinActDist %f (need %d points, have %d points)!\n",
                currentMinActDist, (int)(setting_desiredPointDensity), ef->nPoints);



	FrameHessian* newestHs = frameHessians.back();

	// 2022.1.5
	std::vector<ImmaturePoint*> toOptimize; toOptimize.reserve(cam_num*20000);

	// 2022.1.6
	int immature_invalid_deleted=0,immature_notReady_deleted=0,immature_notReady_skipped=0,immature_deleted=0;

	int cnt = 0;// 2022.1.6
	for(int idx = 0;idx < cam_num;idx ++)
	{
		// make dist map.
		// xtl:设置内参
		coarseDistanceMap->makeK(&Hcalib,idx);
		// xtl:把所有关键帧中的关键点投影至newestHs（最后一个关键帧）上，构建最后一帧的距离地图
		coarseDistanceMap->makeDistanceMap(frameHessians, newestHs, idx);

		//coarseTracker->debugPlotDistMap("distMap");
		// xtl:把所有关键帧中的未成熟点投影到了最新关键帧上，根据刚刚建的距离地图，选择性地激活未成熟点
		for(FrameHessian* host : frameHessians)		// go through all active frames
		{
			if(host == newestHs) continue;

			// 2022.1.2
			SE3 fhToNew = newestHs->frame[idx]->PRE_worldToCam * host->frame[idx]->PRE_camToWorld;
			// xtl:刚刚设置的内参
			Mat33f KRKi = (coarseDistanceMap->K[1] * fhToNew.rotationMatrix().cast<float>() * coarseDistanceMap->Ki[0]);
			Vec3f Kt = (coarseDistanceMap->K[1] * fhToNew.translation().cast<float>());

			// 2022.1.6
			for(unsigned int i=0;i<host->frame[idx]->immaturePoints.size();i+=1)
			{
				ImmaturePoint* ph = host->frame[idx]->immaturePoints[i];
				ph->idxInImmaturePoints = i;

				// delete points that have never been traced successfully, or that are outlier on the last trace.
				// xtl:删除极线搜索失败的点，以及上一次追踪时为外点的点
				if(!std::isfinite(ph->idepth_max) || ph->lastTraceStatus == IPS_OUTLIER)
				{
					// remove point.
					delete ph;
					host->frame[idx]->immaturePoints[i]=0;
					continue;
				}

				// can activate only if this is true.
				bool canActivate = (ph->lastTraceStatus == IPS_GOOD
						|| ph->lastTraceStatus == IPS_SKIPPED
						|| ph->lastTraceStatus == IPS_BADCONDITION
						|| ph->lastTraceStatus == IPS_OOB )
								&& ph->lastTracePixelInterval < 8
								&& ph->quality > setting_minTraceQuality
								&& (ph->idepth_max+ph->idepth_min) > 0;


				// if I cannot activate the point, skip it. Maybe also delete it.
				if(!canActivate)
				{
					// if point will be out afterwards, delete it instead.
					if(ph->host->flaggedForMarginalization || ph->lastTraceStatus == IPS_OOB)//2022.1.6
					{
						delete ph;
						host->frame[idx]->immaturePoints[i]=0;
					}
					continue;
				}


				// see if we need to activate point due to distance map.
				Vec3f ptp = KRKi * Vec3f(ph->u, ph->v, 1) + Kt*(0.5f*(ph->idepth_max+ph->idepth_min));
				int u = ptp[0] / ptp[2] + 0.5f;
				int v = ptp[1] / ptp[2] + 0.5f;

				if((u > 0 && v > 0 && u < wG[1] && v < hG[1])&&(maskG[idx][1][u+v*wG[1]])) //2022.1.4
				{

					float dist = coarseDistanceMap->fwdWarpedIDDistFinal[u+wG[1]*v] + (ptp[0]-floorf((float)(ptp[0])));

					if(dist>=currentMinActDist* ph->my_type)
					{
						coarseDistanceMap->addIntoDistFinal(u,v);
						toOptimize.push_back(ph);
						// 2022.1.6
						if(idx == 1)
							cnt++;//
					}
				}
				else
				{
					delete ph;
					host->frame[idx]->immaturePoints[i]=0;
				}
			}
			// 2021.10.31

		}
	}
	// 2022.1.5
	/*
	printf("Frame[0]:toOptimize: %d. (del %d, notReady %d, invalid %d, skipped %d)\n",
			(int)toOptimize.size()-cnt, immature_deleted, immature_notReady_deleted, immature_invalid_deleted, immature_notReady_skipped);
	*/

	std::vector<PointHessian*> optimized; optimized.resize(toOptimize.size());

	// XTL：构建toOptimize中未成熟点对所有关键帧的残差，优化其逆深度，若好的残差足够（1个）则把该未成熟点放入optimized，并且把残差放入该点的residuals当中
	if(multiThreading)
		treadReduce.reduce(boost::bind(&FullSystem::activatePointsMT_Reductor, this, &optimized, &toOptimize, _1, _2, _3, _4), 0, toOptimize.size(), 50);

	else
		activatePointsMT_Reductor(&optimized, &toOptimize, 0, toOptimize.size(), 0, 0);

	for(unsigned k=0;k<toOptimize.size();k++)
	{
		PointHessian* newpoint = optimized[k];
		ImmaturePoint* ph = toOptimize[k];

		// XTL：若该点优化之后被认为是好的点，则删除原来的未成熟点，并将该点放入主帧的pointHessians当中
		if(newpoint != 0 && newpoint != (PointHessian*)((long)(-1)))
		{
			newpoint->host->immaturePoints[ph->idxInImmaturePoints]=0;
			newpoint->host->pointHessians.push_back(newpoint);
			// 2022.1.6

			ef->insertPoint(newpoint);
			for(PointFrameResidual* r : newpoint->residuals)
				ef->insertResidual(r);
			assert(newpoint->efPoint != 0);
			delete ph;
		}// XTL：若该点优化后被认为是不好的，或者上次追踪状态为OOB时，就将原来的未成熟点删掉
		else if(newpoint == (PointHessian*)((long)(-1)) || ph->lastTraceStatus==IPS_OOB)
		{
			ph->host->immaturePoints[ph->idxInImmaturePoints]=0;
			delete ph;
		}
		else
		{
			assert(newpoint == 0 || newpoint == (PointHessian*)((long)(-1)));
		}
	}
	// 将要删除的点移到immaturePoints末尾，然后pop掉
	for(FrameHessian* host : frameHessians)
	{
		// 2021.11.16
		for(int idx=0;idx<cam_num;idx++){
			for(int i=0;i<(int)host->frame[idx]->immaturePoints.size();i++)
			{
				if(host->frame[idx]->immaturePoints[i]==0)
				{
					host->frame[idx]->immaturePoints[i] = host->frame[idx]->immaturePoints.back();
					host->frame[idx]->immaturePoints.pop_back();
					i--;
				}
			}
		}
	}

}






void FullSystem::activatePointsOldFirst()
{
	assert(false);
}

void FullSystem::flagPointsForRemoval()
{
	assert(EFIndicesValid);

	std::vector<FrameHessian*> fhsToKeepPoints;
	std::vector<FrameHessian*> fhsToMargPoints;

	//if(setting_margPointVisWindow>0)
	{	
		for(int i=((int)frameHessians.size())-1;i>=0 && i >= ((int)frameHessians.size());i--){
			if(!frameHessians[i]->flaggedForMarginalization) fhsToKeepPoints.push_back(frameHessians[i]);
		}
		// XTL：遍历关键帧，若标记为了边缘化帧，就放入fhsToMargPoints当中
		for(int i=0; i< (int)frameHessians.size();i++){
			if(frameHessians[i]->flaggedForMarginalization) 
				// 2021.11.18
				for(int idx=0;idx<cam_num;idx++){
					fhsToMargPoints.push_back(frameHessians[i]->frame[idx]);
				}
				// 2021.11.18
		}
	}

	int flag_oob=0, flag_in=0, flag_inin=0, flag_nores=0;

	// XTL：遍历关键帧上的关键点，若其逆深度小于0或者残差项数量为0,就删掉该点，若边缘化掉关键帧后点残差数量变得过少/该点在最近两帧的追踪状态都较差/该点的主帧被边缘化了，就
	for(FrameHessian* host : frameHessians)		// go through all active frames
	{

		for(unsigned int idx=0;idx<cam_num;idx++)
		{
			for(unsigned int i=0;i<host->frame[idx]->pointHessians.size();i++)
			{
				PointHessian* ph = host->frame[idx]->pointHessians[i];
				if(ph==0) continue;

				if(ph->idepth_scaled < 0 || ph->residuals.size()==0)
				{
					host->frame[idx]->pointHessiansOut.push_back(ph);
					ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
					host->frame[idx]->pointHessians[i]=0;
					flag_nores++;
				}
				else if(ph->isOOB(fhsToKeepPoints, fhsToMargPoints) || host->frame[idx]->flaggedForMarginalization)
				{
					flag_oob++;
					// XTL：若在这些帧还没有边缘化时，这个点残差项数量足够，则遍历这个点的残差项，重置（能量置零，state=IN,New_state=OUTLIER）,并进行残差、导数的计算
					// XTL：若其在计算之后发现点的残差是IN，则将残差项边缘化保留下来
					// XTL：若点的逆深度H矩阵大于阈值，则标记为边缘化的点，否则标记为PS_DROP
					if(ph->isInlierNew())
					{
						flag_in++;
						int ngoodRes=0;
						for(PointFrameResidual* r : ph->residuals)
						{
							r->resetOOB();
							r->linearize(&Hcalib);
							r->efResidual->isLinearized = false;
							r->applyRes(true);
							if(r->efResidual->isActive())
							{
								r->efResidual->fixLinearizationF(ef);
								ngoodRes++;
							}
						}
						if(ph->idepth_hessian > setting_minIdepthH_marg)
						{
							flag_inin++;
							ph->efPoint->stateFlag = EFPointStatus::PS_MARGINALIZE;
							host->frame[idx]->pointHessiansMarginalized.push_back(ph);
						}
						else
						{
							ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
							host->frame[idx]->pointHessiansOut.push_back(ph);
						}


					}
					else
					{
						host->frame[idx]->pointHessiansOut.push_back(ph);
						ph->efPoint->stateFlag = EFPointStatus::PS_DROP;


						//printf("drop point in frame %d (%d goodRes, %d activeRes)\n", ph->host->idx, ph->numGoodResiduals, (int)ph->residuals.size());
					}

					host->frame[idx]->pointHessians[i]=0;
				}
			}

			for(int i=0;i<(int)host->frame[idx]->pointHessians.size();i++)
			{
				if(host->frame[idx]->pointHessians[i]==0)
				{
					host->frame[idx]->pointHessians[i] = host->frame[idx]->pointHessians.back();
					host->frame[idx]->pointHessians.pop_back();
					i--;
				}
			}

		/*
		std::cout<<"sum flag_nores "<<flag_nores<<std::endl;
		std::cout<<"sum flag_oob "<<flag_oob<<std::endl;
		std::cout<<"sum flag_in "<<flag_in<<std::endl;
		std::cout<<"flag host->pointHessians.size()"<<host->pointHessians.size()<<std::endl;
		*/
		// 2021.10.31
		}
	}

}
// 2022.1.8
void draw(cv::Mat& m, int Ku, int Kv)
{
    for(int i = Ku; i <= Ku; i++)
    {
        for(int j = Kv; j <= Kv; j++)
        {
            m.at<unsigned char>(j, i) = 255;
        }
    }
}

void drawMask()
{
	for(int cam_idx=0;cam_idx<cam_num;cam_idx++)
	{
		cv::Mat m(hG[0], wG[0], CV_8UC1, cv::Scalar(0));
		for(int i=0;i<wG[0];i++){
			for(int j=0;j<hG[0];j++){
				if((Var_mask[cam_idx][i+j*wG[0]]-(int)Var_mask[cam_idx][i+j*wG[0]])<0.01) continue;
				draw(m,i,j);
			}
		}
		char str[100];
		sprintf(str, "/home/likun/catkin_ws/src/dso-omni/visualPointDistribution/%d.png",cam_idx);
		cv::imwrite(std::string(str), m);
		m.release();
	}
}//

void FullSystem::addActiveFrame( std::vector<ImageAndExposure*> image, int id )
{

    if(isLost) return;
	boost::unique_lock<boost::mutex> lock(trackMutex);


	// =========================== add into allFrameHistory =========================
	FrameHessian* fh = new FrameHessian();
	FrameShell* shell = new FrameShell();
	shell->camToWorld = SE3(); 		// no lock required, as fh is not used anywhere yet.
	// 2021.12.29
	shell->aff_g2l = AffLight(0,0);
	/*
	for(int i=0;i<cam_num;i++){
		shell->aff_g2l[i] = AffLight(0,0);
	}
	*/
	// std::cout<<"addActiveFrame-a"<<std::endl;
	// 2021.12.29
    shell->marginalizedAt = shell->id = allFrameHistory.size();
    shell->timestamp = image[0]->timestamp;
    shell->incoming_id = id;
	fh->shell = shell;
	allFrameHistory.push_back(shell);


	// =========================== make Images / derivatives etc. =========================
	fh->ab_exposure = image[0]->exposure_time;
    fh->makeImages(image[0]->image, &Hcalib);

	// std::cout<<"addActiveFrame-b"<<std::endl;
	// 2021.12.29
	fh->cam_idx = 0;
	fh->frame[0] = fh;
	for(int i=1;i<cam_num;i++){
		FrameHessian* fh_tmp = new FrameHessian();
		fh_tmp->cam_idx = i;
		fh_tmp->ab_exposure = image[i]->exposure_time;
		fh_tmp->makeImages(image[i]->image, &Hcalib);
		fh_tmp->shell = shell;
		fh_tmp->frame[0] = fh;
		fh->frame[i] = fh_tmp;
	}
	//

	if(!initialized)
	{
		// use initializer!
		if(coarseInitializer->frameID<0)	// first frame set. fh is kept by coarseInitializer.
		{
			coarseInitializer->setFirst(&Hcalib, fh);
		}
		else if(coarseInitializer->trackFrame(fh, outputWrapper))	// if SNAPPED
		{
			initializeFromInitializer(fh);
			lock.unlock();
			deliverTrackedFrame(fh, true);
		}
		else
		{
			// if still initializing
			fh->shell->poseValid = false;
			delete fh;
		}
		return;
	}
	else	// do front-end operation.
	{
		// =========================== SWAP tracking reference?. =========================
		if(coarseTracker_forNewKF->refFrameID > coarseTracker->refFrameID)
		{
			boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);
			CoarseTracker* tmp = coarseTracker; coarseTracker=coarseTracker_forNewKF; coarseTracker_forNewKF=tmp;
		}

		// std::cout<<"addActiveFrame-c"<<std::endl;

		Vec4 tres = trackNewCoarse(fh);
		// std::cout<<"addActiveFrame-d"<<std::endl;
		if(!std::isfinite((double)tres[0]) || !std::isfinite((double)tres[1]) || !std::isfinite((double)tres[2]) || !std::isfinite((double)tres[3]))
        {
            printf("Initial Tracking failed: LOST!\n");
			isLost=true;
            return;
        }

		bool needToMakeKF = false;
		if(setting_keyframesPerSecond > 0)
		{
			needToMakeKF = allFrameHistory.size()== 1 ||
					(fh->shell->timestamp - allKeyFramesHistory.back()->timestamp) > 0.95f/setting_keyframesPerSecond;
		}
		else
		{
			Vec2 refToFh=AffLight::fromToVecExposure(coarseTracker->lastRef->ab_exposure, fh->ab_exposure,
					coarseTracker->lastRef_aff_g2l, fh->shell->aff_g2l);

			// BRIGHTNESS CHECK
			needToMakeKF = allFrameHistory.size()== 1 ||
					setting_kfGlobalWeight*setting_maxShiftWeightT *  sqrtf((double)tres[1]) / (wG[0]+hG[0]) +
					setting_kfGlobalWeight*setting_maxShiftWeightR *  sqrtf((double)tres[2]) / (wG[0]+hG[0]) +
					setting_kfGlobalWeight*setting_maxShiftWeightRT * sqrtf((double)tres[3]) / (wG[0]+hG[0]) +
					setting_kfGlobalWeight*setting_maxAffineWeight * fabs(logf((float)refToFh[0])) > 1 ||
					2*coarseTracker->firstCoarseRMSE < tres[0];

		}




        for(IOWrap::Output3DWrapper* ow : outputWrapper)
            ow->publishCamPose(fh->shell, &Hcalib);




		lock.unlock();
		deliverTrackedFrame(fh, needToMakeKF);
		// 2022.1.8
		drawMask();
		// 2022.1.8
		return;
	}
}
void FullSystem::deliverTrackedFrame(FrameHessian* fh, bool needKF)
{


	if(linearizeOperation)
	{
		if(goStepByStep && lastRefStopID != coarseTracker->refFrameID)
		{
			MinimalImageF3 img(wG[0], hG[0], fh->dI);
			IOWrap::displayImage("frameToTrack", &img);
			while(true)
			{
				char k=IOWrap::waitKey(0);
				if(k==' ') break;
				handleKey( k );
			}
			lastRefStopID = coarseTracker->refFrameID;
		}
		else handleKey( IOWrap::waitKey(1) );



		if(needKF) makeKeyFrame(fh);
		else makeNonKeyFrame(fh);
	}
	else
	{
		boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);
		unmappedTrackedFrames.push_back(fh);
		if(needKF) needNewKFAfter=fh->shell->trackingRef->id;
		trackedFrameSignal.notify_all();

		while(coarseTracker_forNewKF->refFrameID == -1 && coarseTracker->refFrameID == -1 )
		{
			mappedFrameSignal.wait(lock);
		}

		lock.unlock();
	}
}

void FullSystem::mappingLoop()
{
	boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);

	while(runMapping)
	{
		while(unmappedTrackedFrames.size()==0)
		{
			trackedFrameSignal.wait(lock);
			if(!runMapping) return;
		}

		FrameHessian* fh = unmappedTrackedFrames.front();
		unmappedTrackedFrames.pop_front();

		// guaranteed to make a KF for the very first two tracked frames.
		if(allKeyFramesHistory.size() <= 2)
		{
			lock.unlock();
			makeKeyFrame(fh);
			lock.lock();
			mappedFrameSignal.notify_all();
			continue;
		}

		if(unmappedTrackedFrames.size() > 3)
			needToKetchupMapping=true;


		if(unmappedTrackedFrames.size() > 0) // if there are other frames to tracke, do that first.
		{
			lock.unlock();
			makeNonKeyFrame(fh);
			lock.lock();

			if(needToKetchupMapping && unmappedTrackedFrames.size() > 0)
			{
				FrameHessian* fh = unmappedTrackedFrames.front();
				unmappedTrackedFrames.pop_front();
				{
					boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
					assert(fh->shell->trackingRef != 0);
					fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
					fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(),fh->shell->aff_g2l);
				}
				// 2021.12.01
				for(int i=1;i<cam_num;i++){
					delete fh->frame[i];
				}
				// 2021.12.01
				delete fh;
			}

		}
		else
		{
			if(setting_realTimeMaxKF || needNewKFAfter >= frameHessians.back()->shell->id)
			{
				lock.unlock();
				makeKeyFrame(fh);
				needToKetchupMapping=false;
				lock.lock();
			}
			else
			{
				lock.unlock();
				makeNonKeyFrame(fh);
				lock.lock();
			}
		}
		mappedFrameSignal.notify_all();
	}
	printf("MAPPING FINISHED!\n");
}

void FullSystem::blockUntilMappingIsFinished()
{
	boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);
	runMapping = false;
	trackedFrameSignal.notify_all();
	lock.unlock();

	mappingThread.join();

}

void FullSystem::makeNonKeyFrame( FrameHessian* fh)
{
	// needs to be set by mapping thread. no lock required since we are in mapping thread.
	{
		boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
		assert(fh->shell->trackingRef != 0);
		fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
		fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(),fh->shell->aff_g2l);
	}

	traceNewCoarse(fh);
	// 2021.12.29
	for(int i=1;i<cam_num;i++)
		delete fh->frame[i];
	// 2021.12.29
	delete fh;
}

void FullSystem::makeKeyFrame( FrameHessian* fh)
{
	// 设置当前帧的状态量
	// needs to be set by mapping thread
	{
		boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
		assert(fh->shell->trackingRef != 0);
		fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
		fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(),fh->shell->aff_g2l);
	}

	//xtl: 把所有关键帧上的非成熟点投影到fh上，进行极线搜索，根据搜索的情况设置ph的lastTraceStatus
	//若搜索成功了，就设置lastTraceUV为该帧上的坐标，设置lastTracePixelInterval，来表现优化后坐标可能处于的范围大小（不确定度）
	traceNewCoarse(fh);

	boost::unique_lock<boost::mutex> lock(mapMutex);

	// =========================== Flag Frames to be Marginalized. =========================
	// 2021.11.2
	// TODO marginalize can be modified(only marginalize one of the frames)
	// XTL：标记需要边缘化的帧
	flagFramesForMarginalization(fh);
	// 2021.11.2

	// =========================== add New Frame to Hessian Struct. =========================
	fh->idx = frameHessians.size();
	// 2021.11.16
	for(int i=1;i<cam_num;i++){
		fh->frame[i]->idx = frameHessians.size();
		fh->frame[i]->frameID = allKeyFramesHistory.size();
	}
	// 2021.11.16
	frameHessians.push_back(fh);
	fh->frameID = allKeyFramesHistory.size();
	allKeyFramesHistory.push_back(fh->shell);
	ef->insertFrame(fh, &Hcalib);
	// ef->insertFrame(fh->right, &Hcalib);

	setPrecalcValues();



	// =========================== add new residuals for old points =========================
	int numFwdResAdde=0;
	// 遍历所有的关键帧上的关键点，构建点与当前帧的残差项，初始化残差项（state_state为IN，点的lastResiduals前移一个单元，最后一个lastResiduals为新的残差项），PointFrameResidual插入到ef当中
	for(FrameHessian* fh1 : frameHessians)		// go through all active frames
	{
		if(fh1 == fh) continue;
		// 2021.11.15
		for(int i=0;i<cam_num;i++)
		{
			for(PointHessian* ph : fh1->frame[i]->pointHessians)
			{
				PointFrameResidual* r = new PointFrameResidual(ph, fh1->frame[i], fh->frame[i]);
				r->setState(ResState::IN);
				ph->residuals.push_back(r);
				ef->insertResidual(r);
				ph->lastResiduals[1] = ph->lastResiduals[0];
				ph->lastResiduals[0] = std::pair<PointFrameResidual*, ResState>(r, ResState::IN);
				numFwdResAdde+=1;
			}
		}
	}




	// =========================== Activate Points (& flag for marginalization). =========================
	// XTL:激活合适的未成熟点
	activatePointsMT();
	// XTL:重新对ef中的frame进行编号;遍历各个frame上的点，放入allPoints，更新该点所有残差项的hostIDX与targetIDX
	ef->makeIDX();




	// =========================== OPTIMIZE ALL =========================

	fh->frameEnergyTH = frameHessians.back()->frameEnergyTH;
	float rmse = optimize(setting_maxOptIterations);

	// std::cout<<"makeKeyFrame-a"<<std::endl;

	// =========================== Figure Out if INITIALIZATION FAILED =========================
	/*
	if(allKeyFramesHistory.size() <= 4)
	{
		if(allKeyFramesHistory.size()==2 && rmse > 20*benchmark_initializerSlackFactor)
		{
			printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
			initFailed=true;
		}
		if(allKeyFramesHistory.size()==3 && rmse > 13*benchmark_initializerSlackFactor)
		{
			printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
			initFailed=true;
		}
		if(allKeyFramesHistory.size()==4 && rmse > 9*benchmark_initializerSlackFactor)
		{
			printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
			initFailed=true;
		}
	}*/

    if(isLost) return;

	// =========================== REMOVE OUTLIER =========================
	removeOutliers();

	{
		boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);
		// XTL：设置coarseTracker_forNewKF的内参
		coarseTracker_forNewKF->makeK(&Hcalib);
		// std::cout<<"makeKeyFrame-b"<<std::endl;
		// XTL：设置lastRef为最新一个关键帧，refFrameID为该帧的shell的id，将所有与最新关键帧构建了残差的点的投影坐标记录下来，便于可视化
		coarseTracker_forNewKF->setCoarseTrackingRef(frameHessians);
		// std::cout<<"makeKeyFrame-c"<<std::endl;
		// XTL：深度图可视化的一些操作
        coarseTracker_forNewKF->debugPlotIDepthMap(&minIdJetVisTracker, &maxIdJetVisTracker, outputWrapper);
        coarseTracker_forNewKF->debugPlotIDepthMapFloat(outputWrapper);
		// std::cout<<"makeKeyFrame-d"<<std::endl;
	}


	debugPlot("post Optimize");




	// =========================== (Activate-)Marginalize Points =========================
	flagPointsForRemoval();
	// std::cout<<"makeKeyFrame-e"<<std::endl;
	ef->dropPointsF();
	getNullspaces(
			ef->lastNullspaces_pose,
			ef->lastNullspaces_scale,
			ef->lastNullspaces_affA,
			ef->lastNullspaces_affB);
	// std::cout<<"makeKeyFrame-f"<<std::endl;
	ef->marginalizePointsF();

	// std::cout<<"makeKeyFrame-g"<<std::endl;

	// =========================== add new Immature points & new residuals =========================
	makeNewTraces(fh, 0);

	// std::cout<<"makeKeyFrame-h"<<std::endl;

    for(IOWrap::Output3DWrapper* ow : outputWrapper)
    {
        ow->publishGraph(ef->connectivityMap);
        ow->publishKeyframes(frameHessians, false, &Hcalib);
    }



	// =========================== Marginalize Frames =========================

	for(unsigned int i=0;i<frameHessians.size();i++)
		if(frameHessians[i]->flaggedForMarginalization)
			{
				marginalizeFrame(frameHessians[i]); i=0;
			}



	printLogLine();
    //printEigenValLine();
	// std::cout<<"makeKeyFrame-i"<<std::endl;
}


void FullSystem::initializeFromInitializer(FrameHessian* newFrame)
{
	boost::unique_lock<boost::mutex> lock(mapMutex);

	// add firstframe.
	FrameHessian* firstFrame = coarseInitializer->firstFrame;
	firstFrame->idx = frameHessians.size();
	frameHessians.push_back(firstFrame);
	firstFrame->frameID = allKeyFramesHistory.size();
	for(int i=1;i<cam_num;i++){
		firstFrame->frame[i]->idx = firstFrame->idx;
		firstFrame->frame[i]->frameID = firstFrame->frameID;
	}
	allKeyFramesHistory.push_back(firstFrame->shell);
	ef->insertFrame(firstFrame, &Hcalib);
	// 2021.11.2
	// ef->insertFrame(firstFrame_r, &Hcalib);
	// 2021.11.2
	setPrecalcValues();

	//int numPointsTotal = makePixelStatus(firstFrame->dI, selectionMap, wG[0], hG[0], setting_desiredDensity);
	//int numPointsTotal = pixelSelector->makeMaps(firstFrame->dIp, selectionMap,setting_desiredDensity);
	// 2022.1.2
	float TotalsumID=1e-5, TotalnumID=1e-5;
	for(int idx=0;idx<cam_num;idx++)
	{
		firstFrame->frame[idx]->pointHessians.reserve(wG[0]*hG[0]*0.2f);
		firstFrame->frame[idx]->pointHessiansMarginalized.reserve(wG[0]*hG[0]*0.2f);
		firstFrame->frame[idx]->pointHessiansOut.reserve(wG[0]*hG[0]*0.2f);

		float sumID=1e-5, numID=1e-5;
		for(int i=0;i<coarseInitializer->numPoints[idx][0];i++)
		{
			sumID += coarseInitializer->points[idx][0][i].iR;
			numID++;
		}
		TotalsumID+=sumID;
		TotalnumID+=numID;
		float rescaleFactor = 1 / (sumID / numID);
		// randomly sub-select the points I need.
		float keepPercentage = setting_desiredPointDensity / coarseInitializer->numPoints[idx][0];
		
		if(!setting_debugout_runquiet){
			printf("Initialization: keep %.1f%% (need %d, have %d)!\n", 100*keepPercentage,
					(int)(setting_desiredPointDensity), coarseInitializer->numPoints[idx][0]);
		}
		// 2021.11.2

		for(int i=0;i<coarseInitializer->numPoints[idx][0];i++)
		{
			if(rand()/(float)RAND_MAX > keepPercentage) continue;

			Pnt* point = coarseInitializer->points[idx][0]+i;
			ImmaturePoint* pt = new ImmaturePoint(point->u+0.5f,point->v+0.5f,firstFrame->frame[idx],point->my_type, &Hcalib);

			if(!std::isfinite(pt->energyTH)) { delete pt; continue; }


			pt->idepth_max=pt->idepth_min=1;
			PointHessian* ph = new PointHessian(pt, &Hcalib);
			delete pt;
			if(!std::isfinite(ph->energyTH)) {delete ph; continue;}

			ph->setIdepthScaled(point->iR*rescaleFactor);
			ph->setIdepthZero(ph->idepth);
			ph->hasDepthPrior=true;
			ph->setPointStatus(PointHessian::ACTIVE);

			firstFrame->frame[idx]->pointHessians.push_back(ph);
			ef->insertPoint(ph);
		}
	}
	float rescaleFactor = 1 / (TotalsumID / TotalnumID);
	// 2022.1.2
	SE3 firstToNew = coarseInitializer->thisToNext;
	firstToNew.translation() /= rescaleFactor;


	// really no lock required, as we are initializing.
	{
		boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
		firstFrame->shell->camToWorld = SE3();
		firstFrame->shell->aff_g2l = AffLight(0,0);
		firstFrame->setEvalPT_scaled(firstFrame->shell->camToWorld.inverse(),firstFrame->shell->aff_g2l);
		firstFrame->shell->trackingRef=0;
		firstFrame->shell->camToTrackingRef = SE3();

		newFrame->shell->camToWorld = firstToNew.inverse();
		newFrame->shell->aff_g2l = AffLight(0,0);
		newFrame->setEvalPT_scaled(newFrame->shell->camToWorld.inverse(),newFrame->shell->aff_g2l);
		newFrame->shell->trackingRef = firstFrame->shell;
		newFrame->shell->camToTrackingRef = firstToNew.inverse();

	}

	initialized=true;
	// 2022.1.4
	int num_ph = 0;
	for(int i=0;i<cam_num;i++){
		num_ph += firstFrame->frame[i]->pointHessians.size();
	}
	printf("INITIALIZE FROM INITIALIZER (%d pts)!\n", num_ph);
	//
}

// XTL：遍历新帧，选点创建未成熟点
void FullSystem::makeNewTraces(FrameHessian* newFrame, float* gtDepth)
{
	pixelSelector->allowFast = true;

	// 2021.11.9
	for(int idx=0;idx<cam_num;idx++){
		// 2021.11.9
		//int numPointsTotal = makePixelStatus(newFrame->dI, selectionMap, wG[0], hG[0], setting_desiredDensity);
		int numPointsTotal = pixelSelector->makeMaps(newFrame->frame[idx], selectionMap,setting_desiredImmatureDensity);

		newFrame->frame[idx]->pointHessians.reserve(numPointsTotal*1.2f);
		newFrame->frame[idx]->pointHessiansMarginalized.reserve(numPointsTotal*1.2f);
		newFrame->frame[idx]->pointHessiansOut.reserve(numPointsTotal*1.2f);


		for(int y=patternPadding+1;y<hG[0]-patternPadding-2;y++)
		for(int x=patternPadding+1;x<wG[0]-patternPadding-2;x++)
		{
			int i = x+y*wG[0];
			if(selectionMap[i]==0) continue;

			// 2022.1.7 TODO
			// XTL：若取到了和之前取到的点同一个位置且颜色和之前一样的点
			if( (int)Var_mask[idx][i] == (int)newFrame->frame[idx]->dI[i][0] )
			{
				if( (Var_mask[idx][i] - (int)Var_mask[idx][i]) <= 0.999 )
					Var_mask[idx][i] += 0.001;
			}
			else if( Var_mask[idx][i] != 0.0 )// XTL：若Var_mask的某个位置之前已经赋值过了
				Var_mask[idx][i] -= 0.001;
			if( Var_mask[idx][i] == 0 )
				Var_mask[idx][i] = ((int)newFrame->frame[idx]->dI[i][0]*1.0);
			///

			ImmaturePoint* impt = new ImmaturePoint(x,y,newFrame->frame[idx], selectionMap[i], &Hcalib);
			if(!std::isfinite(impt->energyTH)) delete impt;
			else newFrame->frame[idx]->immaturePoints.push_back(impt);

		}
		// 2022.1.5
		/*
		printf("MADE %d IMMATURE POINTS in %d'th cam!\n", (int)newFrame->frame[idx]->immaturePoints.size() , idx);
		*/
	}

}
/*
void draw(cv::Mat& m, int Ku, int Kv)
{
    for(int i = Ku; i <= Ku; i++)
    {
        for(int j = Kv; j <= Kv; j++)
        {
            m.at<unsigned char>(j, i) = 255;
        }
    }
}*/

void FullSystem::setPrecalcValues()
{
	for(FrameHessian* fh : frameHessians)
	{
		fh->targetPrecalc.resize(frameHessians.size());
		for(unsigned int i=0;i<frameHessians.size();i++){
			fh->targetPrecalc[i].set(fh, frameHessians[i], &Hcalib);
		}
	}

	ef->setDeltaF(&Hcalib);
}


void FullSystem::printLogLine()
{
	if(frameHessians.size()==0) return;

    if(!setting_debugout_runquiet)
        printf("LOG %d: %.3f fine. Res: %d A, %d L, %d M; (%'d / %'d) forceDrop. a=%f, b=%f. Window %d (%d)\n",
                allKeyFramesHistory.back()->id,
                statistics_lastFineTrackRMSE,
                ef->resInA,
                ef->resInL,
                ef->resInM,
                (int)statistics_numForceDroppedResFwd,
                (int)statistics_numForceDroppedResBwd,
                allKeyFramesHistory.back()->aff_g2l.a,
                allKeyFramesHistory.back()->aff_g2l.b,
                frameHessians.back()->shell->id - frameHessians.front()->shell->id,
                (int)frameHessians.size());


	if(!setting_logStuff) return;

	if(numsLog != 0)
	{
		(*numsLog) << allKeyFramesHistory.back()->id << " "  <<
				statistics_lastFineTrackRMSE << " "  <<
				(int)statistics_numCreatedPoints << " "  <<
				(int)statistics_numActivatedPoints << " "  <<
				(int)statistics_numDroppedPoints << " "  <<
				(int)statistics_lastNumOptIts << " "  <<
				ef->resInA << " "  <<
				ef->resInL << " "  <<
				ef->resInM << " "  <<
				statistics_numMargResFwd << " "  <<
				statistics_numMargResBwd << " "  <<
				statistics_numForceDroppedResFwd << " "  <<
				statistics_numForceDroppedResBwd << " "  <<
				frameHessians.back()->aff_g2l().a << " "  <<
				frameHessians.back()->aff_g2l().b << " "  <<
				frameHessians.back()->shell->id - frameHessians.front()->shell->id << " "  <<
				(int)frameHessians.size() << " "  << "\n";
		numsLog->flush();
	}


}



void FullSystem::printEigenValLine()
{
	if(!setting_logStuff) return;
	if(ef->lastHS.rows() < 12) return;


	MatXX Hp = ef->lastHS.bottomRightCorner(ef->lastHS.cols()-CPARS,ef->lastHS.cols()-CPARS);
	MatXX Ha = ef->lastHS.bottomRightCorner(ef->lastHS.cols()-CPARS,ef->lastHS.cols()-CPARS);
	int n = Hp.cols()/8;
	assert(Hp.cols()%8==0);

	// sub-select
	for(int i=0;i<n;i++)
	{
		MatXX tmp6 = Hp.block(i*8,0,6,n*8);
		Hp.block(i*6,0,6,n*8) = tmp6;

		MatXX tmp2 = Ha.block(i*8+6,0,2,n*8);
		Ha.block(i*2,0,2,n*8) = tmp2;
	}
	for(int i=0;i<n;i++)
	{
		MatXX tmp6 = Hp.block(0,i*8,n*8,6);
		Hp.block(0,i*6,n*8,6) = tmp6;

		MatXX tmp2 = Ha.block(0,i*8+6,n*8,2);
		Ha.block(0,i*2,n*8,2) = tmp2;
	}

	VecX eigenvaluesAll = ef->lastHS.eigenvalues().real();
	VecX eigenP = Hp.topLeftCorner(n*6,n*6).eigenvalues().real();
	VecX eigenA = Ha.topLeftCorner(n*2,n*2).eigenvalues().real();
	VecX diagonal = ef->lastHS.diagonal();

	std::sort(eigenvaluesAll.data(), eigenvaluesAll.data()+eigenvaluesAll.size());
	std::sort(eigenP.data(), eigenP.data()+eigenP.size());
	std::sort(eigenA.data(), eigenA.data()+eigenA.size());

	int nz = std::max(100,setting_maxFrames*10);

	if(eigenAllLog != 0)
	{
		VecX ea = VecX::Zero(nz); ea.head(eigenvaluesAll.size()) = eigenvaluesAll;
		(*eigenAllLog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
		eigenAllLog->flush();
	}
	if(eigenALog != 0)
	{
		VecX ea = VecX::Zero(nz); ea.head(eigenA.size()) = eigenA;
		(*eigenALog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
		eigenALog->flush();
	}
	if(eigenPLog != 0)
	{
		VecX ea = VecX::Zero(nz); ea.head(eigenP.size()) = eigenP;
		(*eigenPLog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
		eigenPLog->flush();
	}

	if(DiagonalLog != 0)
	{
		VecX ea = VecX::Zero(nz); ea.head(diagonal.size()) = diagonal;
		(*DiagonalLog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
		DiagonalLog->flush();
	}

	if(variancesLog != 0)
	{
		VecX ea = VecX::Zero(nz); ea.head(diagonal.size()) = ef->lastHS.inverse().diagonal();
		(*variancesLog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
		variancesLog->flush();
	}

	std::vector<VecX> &nsp = ef->lastNullspaces_forLogging;
	(*nullspacesLog) << allKeyFramesHistory.back()->id << " ";
	for(unsigned int i=0;i<nsp.size();i++)
		(*nullspacesLog) << nsp[i].dot(ef->lastHS * nsp[i]) << " " << nsp[i].dot(ef->lastbS) << " " ;
	(*nullspacesLog) << "\n";
	nullspacesLog->flush();

}

void FullSystem::printFrameLifetimes()
{
	if(!setting_logStuff) return;


	boost::unique_lock<boost::mutex> lock(trackMutex);

	std::ofstream* lg = new std::ofstream();
	lg->open("logs/lifetimeLog.txt", std::ios::trunc | std::ios::out);
	lg->precision(15);

	for(FrameShell* s : allFrameHistory)
	{
		(*lg) << s->id
			<< " " << s->marginalizedAt
			<< " " << s->statistics_goodResOnThis
			<< " " << s->statistics_outlierResOnThis
			<< " " << s->movedByOpt;



		(*lg) << "\n";
	}





	lg->close();
	delete lg;

}


void FullSystem::printEvalLine()
{
	return;
}





}
