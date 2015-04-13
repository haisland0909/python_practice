

"""  performs FAST corner detection.
	
	 If you use this in published work, please cite:
	   Faster and better: A machine learning approach to corner detection, E. Rosten, R. Porter and T. Drummond, PAMI 2010
	   Machine learning for high-speed corner detection, E. Rosten and T. Drummond, ECCV 2006
	 The Bibtex entries are:

	 @inproceedings{rosten_2006_machine,
		title       =    "Machine learning for high-speed corner detection",
		author      =    "Edward Rosten and Tom Drummond",
		year        =    "2006",
		month       =    "May",
		pages       =    "430--443",
		volume      =    "1",
		doi         =    "10.1007/11744023_34",
		booktitle   =    "European Conference on Computer Vision",
		notes       =    "Poster presentation",
		url         =    "http://mi.eng.cam.ac.uk/~er258/work/rosten_2006_machine.pdf",
	}

	@article{rosten_2008_faster,
		title       =    "FASTER and better: A machine learning approach to corner detection",
		author      =    "Edward Rosten and Reid Porter and Tom Drummond",
		year        =    "2010",
		journal     =    "IEEE Trans. Pattern Analysis and Machine Intelligence",
		pages       =    "105--119",
		volume      =    "32",
		doi         =    "10.1109/TPAMI.2008.275",
		eprint      =    "arXiv:0810.2434 [cs.CV]",
		url         =    "http://lanl.arXiv.org/pdf/0810.2434",
	}
"""

import numpy

def detect(image, threshold, do_nonmax=1):
	"""
	(corners, scores) = fast9.detect(image, threshold) performs the detection
	on the image and returns the corners as a list of (x,y) tuples and the
	scored as a list of integers.  The score is computed using binary search
	over all possible thresholds.

	(corners, scores) = fast9.detect(image, threshold, nonmax=0) performs the corner
	detection but suppresses nonmaximal suppression.
	"""

	corners = []
	scores=[]
	rows = image.shape[0]
	cols = image.shape[1]

	for y in xrange(4, rows-4):
		for x in xrange(4, cols-4):
			cb = image[y][x] + threshold
			c_b = image[y][x] - threshold
			if image[y+3][x+0] > cb: 
			 if image[y+3][x+1] > cb: 
			  if image[y+2][x+2] > cb: 
			   if image[y+1][x+3] > cb: 
			    if image[y+0][x+3] > cb: 
			     if image[y+-1][x+3] > cb: 
			      if image[y+-2][x+2] > cb: 
			       if image[y+-3][x+1] > cb: 
			        if image[y+-3][x+0] > cb: 
			         pass
			        else:
			         if image[y+3][x+-1] > cb: 
			          pass
			         else:
			          continue
			       elif image[y+-3][x+1] < c_b: 
			        if image[y+2][x+-2] > cb: 
			         if image[y+3][x+-1] > cb: 
			          pass
			         else:
			          continue
			        elif image[y+2][x+-2] < c_b: 
			         if image[y+-3][x+0] < c_b:
			          if image[y+-3][x+-1] < c_b:
			           if image[y+-2][x+-2] < c_b:
			            if image[y+-1][x+-3] < c_b:
			             if image[y+0][x+-3] < c_b:
			              if image[y+1][x+-3] < c_b:
			               if image[y+3][x+-1] < c_b:
			                pass
			               else:
			                continue
			              else:
			               continue
			             else:
			              continue
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			       else:
			        if image[y+2][x+-2] > cb: 
			         if image[y+3][x+-1] > cb: 
			          pass
			         else:
			          continue
			        else:
			         continue
			      elif image[y+-2][x+2] < c_b: 
			       if image[y+3][x+-1] > cb: 
			        if image[y+1][x+-3] > cb: 
			         if image[y+2][x+-2] > cb: 
			          pass
			         else:
			          continue
			        elif image[y+1][x+-3] < c_b: 
			         if image[y+-3][x+1] < c_b:
			          if image[y+-3][x+0] < c_b:
			           if image[y+-3][x+-1] < c_b:
			            if image[y+-2][x+-2] < c_b:
			             if image[y+-1][x+-3] < c_b:
			              if image[y+0][x+-3] < c_b:
			               if image[y+2][x+-2] < c_b:
			                pass
			               else:
			                continue
			              else:
			               continue
			             else:
			              continue
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			       else:
			        if image[y+-3][x+1] < c_b:
			         if image[y+-3][x+0] < c_b:
			          if image[y+-3][x+-1] < c_b:
			           if image[y+-2][x+-2] < c_b:
			            if image[y+-1][x+-3] < c_b:
			             if image[y+0][x+-3] < c_b:
			              if image[y+1][x+-3] < c_b:
			               if image[y+2][x+-2] < c_b:
			                pass
			               else:
			                continue
			              else:
			               continue
			             else:
			              continue
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			      else:
			       if image[y+1][x+-3] > cb: 
			        if image[y+2][x+-2] > cb: 
			         if image[y+3][x+-1] > cb: 
			          pass
			         else:
			          continue
			        else:
			         continue
			       elif image[y+1][x+-3] < c_b: 
			        if image[y+-3][x+1] < c_b:
			         if image[y+-3][x+0] < c_b:
			          if image[y+-3][x+-1] < c_b:
			           if image[y+-2][x+-2] < c_b:
			            if image[y+-1][x+-3] < c_b:
			             if image[y+0][x+-3] < c_b:
			              if image[y+2][x+-2] < c_b:
			               if image[y+3][x+-1] < c_b:
			                pass
			               else:
			                continue
			              else:
			               continue
			             else:
			              continue
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			       else:
			        continue
			     elif image[y+-1][x+3] < c_b: 
			      if image[y+2][x+-2] > cb: 
			       if image[y+0][x+-3] > cb: 
			        if image[y+1][x+-3] > cb: 
			         if image[y+3][x+-1] > cb: 
			          pass
			         else:
			          if image[y+-2][x+2] > cb: 
			           if image[y+-3][x+1] > cb: 
			            if image[y+-3][x+0] > cb: 
			             if image[y+-3][x+-1] > cb: 
			              if image[y+-2][x+-2] > cb: 
			               if image[y+-1][x+-3] > cb: 
			                pass
			               else:
			                continue
			              else:
			               continue
			             else:
			              continue
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			        else:
			         continue
			       elif image[y+0][x+-3] < c_b: 
			        if image[y+-2][x+2] < c_b:
			         if image[y+-3][x+1] < c_b:
			          if image[y+-3][x+0] < c_b:
			           if image[y+-3][x+-1] < c_b:
			            if image[y+-2][x+-2] < c_b:
			             if image[y+-1][x+-3] < c_b:
			              if image[y+1][x+-3] < c_b:
			               pass
			              else:
			               continue
			             else:
			              continue
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			       else:
			        continue
			      elif image[y+2][x+-2] < c_b: 
			       if image[y+-3][x+1] < c_b:
			        if image[y+-3][x+0] < c_b:
			         if image[y+-3][x+-1] < c_b:
			          if image[y+-2][x+-2] < c_b:
			           if image[y+-1][x+-3] < c_b:
			            if image[y+0][x+-3] < c_b:
			             if image[y+1][x+-3] < c_b:
			              if image[y+-2][x+2] < c_b:
			               pass
			              else:
			               if image[y+3][x+-1] < c_b:
			                pass
			               else:
			                continue
			             else:
			              continue
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			       else:
			        continue
			      else:
			       if image[y+-2][x+2] < c_b:
			        if image[y+-3][x+1] < c_b:
			         if image[y+-3][x+0] < c_b:
			          if image[y+-3][x+-1] < c_b:
			           if image[y+-2][x+-2] < c_b:
			            if image[y+-1][x+-3] < c_b:
			             if image[y+0][x+-3] < c_b:
			              if image[y+1][x+-3] < c_b:
			               pass
			              else:
			               continue
			             else:
			              continue
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			       else:
			        continue
			     else:
			      if image[y+0][x+-3] > cb: 
			       if image[y+1][x+-3] > cb: 
			        if image[y+2][x+-2] > cb: 
			         if image[y+3][x+-1] > cb: 
			          pass
			         else:
			          if image[y+-2][x+2] > cb: 
			           if image[y+-3][x+1] > cb: 
			            if image[y+-3][x+0] > cb: 
			             if image[y+-3][x+-1] > cb: 
			              if image[y+-2][x+-2] > cb: 
			               if image[y+-1][x+-3] > cb: 
			                pass
			               else:
			                continue
			              else:
			               continue
			             else:
			              continue
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			        else:
			         continue
			       else:
			        continue
			      elif image[y+0][x+-3] < c_b: 
			       if image[y+-3][x+1] < c_b:
			        if image[y+-3][x+0] < c_b:
			         if image[y+-3][x+-1] < c_b:
			          if image[y+-2][x+-2] < c_b:
			           if image[y+-1][x+-3] < c_b:
			            if image[y+1][x+-3] < c_b:
			             if image[y+2][x+-2] < c_b:
			              if image[y+-2][x+2] < c_b:
			               pass
			              else:
			               if image[y+3][x+-1] < c_b:
			                pass
			               else:
			                continue
			             else:
			              continue
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			       else:
			        continue
			      else:
			       continue
			    elif image[y+0][x+3] < c_b: 
			     if image[y+1][x+-3] > cb: 
			      if image[y+-1][x+-3] > cb: 
			       if image[y+0][x+-3] > cb: 
			        if image[y+2][x+-2] > cb: 
			         if image[y+3][x+-1] > cb: 
			          pass
			         else:
			          if image[y+-2][x+2] > cb: 
			           if image[y+-3][x+1] > cb: 
			            if image[y+-3][x+0] > cb: 
			             if image[y+-3][x+-1] > cb: 
			              if image[y+-2][x+-2] > cb: 
			               pass
			              else:
			               continue
			             else:
			              continue
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			        else:
			         if image[y+-1][x+3] > cb: 
			          if image[y+-2][x+2] > cb: 
			           if image[y+-3][x+1] > cb: 
			            if image[y+-3][x+0] > cb: 
			             if image[y+-3][x+-1] > cb: 
			              if image[y+-2][x+-2] > cb: 
			               pass
			              else:
			               continue
			             else:
			              continue
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			       else:
			        continue
			      elif image[y+-1][x+-3] < c_b: 
			       if image[y+-1][x+3] < c_b:
			        if image[y+-2][x+2] < c_b:
			         if image[y+-3][x+1] < c_b:
			          if image[y+-3][x+0] < c_b:
			           if image[y+-3][x+-1] < c_b:
			            if image[y+-2][x+-2] < c_b:
			             if image[y+0][x+-3] < c_b:
			              pass
			             else:
			              continue
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			       else:
			        continue
			      else:
			       continue
			     elif image[y+1][x+-3] < c_b: 
			      if image[y+-3][x+1] < c_b:
			       if image[y+-3][x+0] < c_b:
			        if image[y+-3][x+-1] < c_b:
			         if image[y+-2][x+-2] < c_b:
			          if image[y+-1][x+-3] < c_b:
			           if image[y+0][x+-3] < c_b:
			            if image[y+-2][x+2] < c_b:
			             if image[y+-1][x+3] < c_b:
			              pass
			             else:
			              if image[y+2][x+-2] < c_b:
			               pass
			              else:
			               continue
			            else:
			             if image[y+2][x+-2] < c_b:
			              if image[y+3][x+-1] < c_b:
			               pass
			              else:
			               continue
			             else:
			              continue
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			       else:
			        continue
			      else:
			       continue
			     else:
			      if image[y+-1][x+3] < c_b:
			       if image[y+-2][x+2] < c_b:
			        if image[y+-3][x+1] < c_b:
			         if image[y+-3][x+0] < c_b:
			          if image[y+-3][x+-1] < c_b:
			           if image[y+-2][x+-2] < c_b:
			            if image[y+-1][x+-3] < c_b:
			             if image[y+0][x+-3] < c_b:
			              pass
			             else:
			              continue
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			       else:
			        continue
			      else:
			       continue
			    else:
			     if image[y+-1][x+-3] > cb: 
			      if image[y+0][x+-3] > cb: 
			       if image[y+1][x+-3] > cb: 
			        if image[y+2][x+-2] > cb: 
			         if image[y+3][x+-1] > cb: 
			          pass
			         else:
			          if image[y+-2][x+2] > cb: 
			           if image[y+-3][x+1] > cb: 
			            if image[y+-3][x+0] > cb: 
			             if image[y+-3][x+-1] > cb: 
			              if image[y+-2][x+-2] > cb: 
			               pass
			              else:
			               continue
			             else:
			              continue
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			        else:
			         if image[y+-1][x+3] > cb: 
			          if image[y+-2][x+2] > cb: 
			           if image[y+-3][x+1] > cb: 
			            if image[y+-3][x+0] > cb: 
			             if image[y+-3][x+-1] > cb: 
			              if image[y+-2][x+-2] > cb: 
			               pass
			              else:
			               continue
			             else:
			              continue
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			       else:
			        continue
			      else:
			       continue
			     elif image[y+-1][x+-3] < c_b: 
			      if image[y+-3][x+1] < c_b:
			       if image[y+-3][x+0] < c_b:
			        if image[y+-3][x+-1] < c_b:
			         if image[y+-2][x+-2] < c_b:
			          if image[y+0][x+-3] < c_b:
			           if image[y+1][x+-3] < c_b:
			            if image[y+-2][x+2] < c_b:
			             if image[y+-1][x+3] < c_b:
			              pass
			             else:
			              if image[y+2][x+-2] < c_b:
			               pass
			              else:
			               continue
			            else:
			             if image[y+2][x+-2] < c_b:
			              if image[y+3][x+-1] < c_b:
			               pass
			              else:
			               continue
			             else:
			              continue
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			       else:
			        continue
			      else:
			       continue
			     else:
			      continue
			   elif image[y+1][x+3] < c_b: 
			    if image[y+-2][x+-2] > cb: 
			     if image[y+-1][x+-3] > cb: 
			      if image[y+0][x+-3] > cb: 
			       if image[y+1][x+-3] > cb: 
			        if image[y+2][x+-2] > cb: 
			         if image[y+3][x+-1] > cb: 
			          pass
			         else:
			          if image[y+-2][x+2] > cb: 
			           if image[y+-3][x+1] > cb: 
			            if image[y+-3][x+0] > cb: 
			             if image[y+-3][x+-1] > cb: 
			              pass
			             else:
			              continue
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			        else:
			         if image[y+-1][x+3] > cb: 
			          if image[y+-2][x+2] > cb: 
			           if image[y+-3][x+1] > cb: 
			            if image[y+-3][x+0] > cb: 
			             if image[y+-3][x+-1] > cb: 
			              pass
			             else:
			              continue
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			       else:
			        if image[y+0][x+3] > cb: 
			         if image[y+-1][x+3] > cb: 
			          if image[y+-2][x+2] > cb: 
			           if image[y+-3][x+1] > cb: 
			            if image[y+-3][x+0] > cb: 
			             if image[y+-3][x+-1] > cb: 
			              pass
			             else:
			              continue
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			      else:
			       continue
			     else:
			      continue
			    elif image[y+-2][x+-2] < c_b: 
			     if image[y+-3][x+1] < c_b:
			      if image[y+-3][x+0] < c_b:
			       if image[y+-3][x+-1] < c_b:
			        if image[y+-1][x+-3] < c_b:
			         if image[y+-2][x+2] < c_b:
			          if image[y+-1][x+3] < c_b:
			           if image[y+0][x+3] < c_b:
			            pass
			           else:
			            if image[y+0][x+-3] < c_b:
			             if image[y+1][x+-3] < c_b:
			              pass
			             else:
			              continue
			            else:
			             continue
			          else:
			           if image[y+0][x+-3] < c_b:
			            if image[y+1][x+-3] < c_b:
			             if image[y+2][x+-2] < c_b:
			              pass
			             else:
			              continue
			            else:
			             continue
			           else:
			            continue
			         else:
			          if image[y+0][x+-3] < c_b:
			           if image[y+1][x+-3] < c_b:
			            if image[y+2][x+-2] < c_b:
			             if image[y+3][x+-1] < c_b:
			              pass
			             else:
			              continue
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			        else:
			         continue
			       else:
			        continue
			      else:
			       continue
			     else:
			      continue
			    else:
			     continue
			   else:
			    if image[y+-2][x+-2] > cb: 
			     if image[y+-1][x+-3] > cb: 
			      if image[y+0][x+-3] > cb: 
			       if image[y+1][x+-3] > cb: 
			        if image[y+2][x+-2] > cb: 
			         if image[y+3][x+-1] > cb: 
			          pass
			         else:
			          if image[y+-2][x+2] > cb: 
			           if image[y+-3][x+1] > cb: 
			            if image[y+-3][x+0] > cb: 
			             if image[y+-3][x+-1] > cb: 
			              pass
			             else:
			              continue
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			        else:
			         if image[y+-1][x+3] > cb: 
			          if image[y+-2][x+2] > cb: 
			           if image[y+-3][x+1] > cb: 
			            if image[y+-3][x+0] > cb: 
			             if image[y+-3][x+-1] > cb: 
			              pass
			             else:
			              continue
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			       else:
			        if image[y+0][x+3] > cb: 
			         if image[y+-1][x+3] > cb: 
			          if image[y+-2][x+2] > cb: 
			           if image[y+-3][x+1] > cb: 
			            if image[y+-3][x+0] > cb: 
			             if image[y+-3][x+-1] > cb: 
			              pass
			             else:
			              continue
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			      else:
			       continue
			     else:
			      continue
			    elif image[y+-2][x+-2] < c_b: 
			     if image[y+-3][x+1] < c_b:
			      if image[y+-3][x+0] < c_b:
			       if image[y+-3][x+-1] < c_b:
			        if image[y+-1][x+-3] < c_b:
			         if image[y+0][x+-3] < c_b:
			          if image[y+-2][x+2] < c_b:
			           if image[y+-1][x+3] < c_b:
			            if image[y+0][x+3] < c_b:
			             pass
			            else:
			             if image[y+1][x+-3] < c_b:
			              pass
			             else:
			              continue
			           else:
			            if image[y+1][x+-3] < c_b:
			             if image[y+2][x+-2] < c_b:
			              pass
			             else:
			              continue
			            else:
			             continue
			          else:
			           if image[y+1][x+-3] < c_b:
			            if image[y+2][x+-2] < c_b:
			             if image[y+3][x+-1] < c_b:
			              pass
			             else:
			              continue
			            else:
			             continue
			           else:
			            continue
			         else:
			          continue
			        else:
			         continue
			       else:
			        continue
			      else:
			       continue
			     else:
			      continue
			    else:
			     continue
			  elif image[y+2][x+2] < c_b: 
			   if image[y+-3][x+-1] > cb: 
			    if image[y+-2][x+-2] > cb: 
			     if image[y+-1][x+-3] > cb: 
			      if image[y+0][x+-3] > cb: 
			       if image[y+1][x+-3] > cb: 
			        if image[y+2][x+-2] > cb: 
			         if image[y+3][x+-1] > cb: 
			          pass
			         else:
			          if image[y+-2][x+2] > cb: 
			           if image[y+-3][x+1] > cb: 
			            if image[y+-3][x+0] > cb: 
			             pass
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			        else:
			         if image[y+-1][x+3] > cb: 
			          if image[y+-2][x+2] > cb: 
			           if image[y+-3][x+1] > cb: 
			            if image[y+-3][x+0] > cb: 
			             pass
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			       else:
			        if image[y+0][x+3] > cb: 
			         if image[y+-1][x+3] > cb: 
			          if image[y+-2][x+2] > cb: 
			           if image[y+-3][x+1] > cb: 
			            if image[y+-3][x+0] > cb: 
			             pass
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			      else:
			       if image[y+1][x+3] > cb: 
			        if image[y+0][x+3] > cb: 
			         if image[y+-1][x+3] > cb: 
			          if image[y+-2][x+2] > cb: 
			           if image[y+-3][x+1] > cb: 
			            if image[y+-3][x+0] > cb: 
			             pass
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			       else:
			        continue
			     else:
			      continue
			    else:
			     continue
			   elif image[y+-3][x+-1] < c_b: 
			    if image[y+-3][x+1] < c_b:
			     if image[y+-3][x+0] < c_b:
			      if image[y+-2][x+-2] < c_b:
			       if image[y+-2][x+2] < c_b:
			        if image[y+-1][x+3] < c_b:
			         if image[y+0][x+3] < c_b:
			          if image[y+1][x+3] < c_b:
			           pass
			          else:
			           if image[y+-1][x+-3] < c_b:
			            if image[y+0][x+-3] < c_b:
			             pass
			            else:
			             continue
			           else:
			            continue
			         else:
			          if image[y+-1][x+-3] < c_b:
			           if image[y+0][x+-3] < c_b:
			            if image[y+1][x+-3] < c_b:
			             pass
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			        else:
			         if image[y+-1][x+-3] < c_b:
			          if image[y+0][x+-3] < c_b:
			           if image[y+1][x+-3] < c_b:
			            if image[y+2][x+-2] < c_b:
			             pass
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			       else:
			        if image[y+-1][x+-3] < c_b:
			         if image[y+0][x+-3] < c_b:
			          if image[y+1][x+-3] < c_b:
			           if image[y+2][x+-2] < c_b:
			            if image[y+3][x+-1] < c_b:
			             pass
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			      else:
			       continue
			     else:
			      continue
			    else:
			     continue
			   else:
			    continue
			  else:
			   if image[y+-3][x+-1] > cb: 
			    if image[y+-2][x+-2] > cb: 
			     if image[y+-1][x+-3] > cb: 
			      if image[y+0][x+-3] > cb: 
			       if image[y+1][x+-3] > cb: 
			        if image[y+2][x+-2] > cb: 
			         if image[y+3][x+-1] > cb: 
			          pass
			         else:
			          if image[y+-2][x+2] > cb: 
			           if image[y+-3][x+1] > cb: 
			            if image[y+-3][x+0] > cb: 
			             pass
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			        else:
			         if image[y+-1][x+3] > cb: 
			          if image[y+-2][x+2] > cb: 
			           if image[y+-3][x+1] > cb: 
			            if image[y+-3][x+0] > cb: 
			             pass
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			       else:
			        if image[y+0][x+3] > cb: 
			         if image[y+-1][x+3] > cb: 
			          if image[y+-2][x+2] > cb: 
			           if image[y+-3][x+1] > cb: 
			            if image[y+-3][x+0] > cb: 
			             pass
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			      else:
			       if image[y+1][x+3] > cb: 
			        if image[y+0][x+3] > cb: 
			         if image[y+-1][x+3] > cb: 
			          if image[y+-2][x+2] > cb: 
			           if image[y+-3][x+1] > cb: 
			            if image[y+-3][x+0] > cb: 
			             pass
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			       else:
			        continue
			     else:
			      continue
			    else:
			     continue
			   elif image[y+-3][x+-1] < c_b: 
			    if image[y+-3][x+1] < c_b:
			     if image[y+-3][x+0] < c_b:
			      if image[y+-2][x+-2] < c_b:
			       if image[y+-1][x+-3] < c_b:
			        if image[y+-2][x+2] < c_b:
			         if image[y+-1][x+3] < c_b:
			          if image[y+0][x+3] < c_b:
			           if image[y+1][x+3] < c_b:
			            pass
			           else:
			            if image[y+0][x+-3] < c_b:
			             pass
			            else:
			             continue
			          else:
			           if image[y+0][x+-3] < c_b:
			            if image[y+1][x+-3] < c_b:
			             pass
			            else:
			             continue
			           else:
			            continue
			         else:
			          if image[y+0][x+-3] < c_b:
			           if image[y+1][x+-3] < c_b:
			            if image[y+2][x+-2] < c_b:
			             pass
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			        else:
			         if image[y+0][x+-3] < c_b:
			          if image[y+1][x+-3] < c_b:
			           if image[y+2][x+-2] < c_b:
			            if image[y+3][x+-1] < c_b:
			             pass
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			       else:
			        continue
			      else:
			       continue
			     else:
			      continue
			    else:
			     continue
			   else:
			    continue
			 elif image[y+3][x+1] < c_b: 
			  if image[y+-3][x+0] > cb: 
			   if image[y+-3][x+-1] > cb: 
			    if image[y+-2][x+-2] > cb: 
			     if image[y+-1][x+-3] > cb: 
			      if image[y+0][x+-3] > cb: 
			       if image[y+1][x+-3] > cb: 
			        if image[y+2][x+-2] > cb: 
			         if image[y+3][x+-1] > cb: 
			          pass
			         else:
			          if image[y+-2][x+2] > cb: 
			           if image[y+-3][x+1] > cb: 
			            pass
			           else:
			            continue
			          else:
			           continue
			        else:
			         if image[y+-1][x+3] > cb: 
			          if image[y+-2][x+2] > cb: 
			           if image[y+-3][x+1] > cb: 
			            pass
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			       else:
			        if image[y+0][x+3] > cb: 
			         if image[y+-1][x+3] > cb: 
			          if image[y+-2][x+2] > cb: 
			           if image[y+-3][x+1] > cb: 
			            pass
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			      else:
			       if image[y+1][x+3] > cb: 
			        if image[y+0][x+3] > cb: 
			         if image[y+-1][x+3] > cb: 
			          if image[y+-2][x+2] > cb: 
			           if image[y+-3][x+1] > cb: 
			            pass
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			       else:
			        continue
			     else:
			      if image[y+2][x+2] > cb: 
			       if image[y+1][x+3] > cb: 
			        if image[y+0][x+3] > cb: 
			         if image[y+-1][x+3] > cb: 
			          if image[y+-2][x+2] > cb: 
			           if image[y+-3][x+1] > cb: 
			            pass
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			       else:
			        continue
			      else:
			       continue
			    else:
			     continue
			   else:
			    continue
			  elif image[y+-3][x+0] < c_b: 
			   if image[y+-3][x+1] < c_b:
			    if image[y+-3][x+-1] < c_b:
			     if image[y+-2][x+2] < c_b:
			      if image[y+-1][x+3] < c_b:
			       if image[y+0][x+3] < c_b:
			        if image[y+1][x+3] < c_b:
			         if image[y+2][x+2] < c_b:
			          pass
			         else:
			          if image[y+-2][x+-2] < c_b:
			           if image[y+-1][x+-3] < c_b:
			            pass
			           else:
			            continue
			          else:
			           continue
			        else:
			         if image[y+-2][x+-2] < c_b:
			          if image[y+-1][x+-3] < c_b:
			           if image[y+0][x+-3] < c_b:
			            pass
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			       else:
			        if image[y+-2][x+-2] < c_b:
			         if image[y+-1][x+-3] < c_b:
			          if image[y+0][x+-3] < c_b:
			           if image[y+1][x+-3] < c_b:
			            pass
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			      else:
			       if image[y+-2][x+-2] < c_b:
			        if image[y+-1][x+-3] < c_b:
			         if image[y+0][x+-3] < c_b:
			          if image[y+1][x+-3] < c_b:
			           if image[y+2][x+-2] < c_b:
			            pass
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			       else:
			        continue
			     else:
			      if image[y+-2][x+-2] < c_b:
			       if image[y+-1][x+-3] < c_b:
			        if image[y+0][x+-3] < c_b:
			         if image[y+1][x+-3] < c_b:
			          if image[y+2][x+-2] < c_b:
			           if image[y+3][x+-1] < c_b:
			            pass
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			       else:
			        continue
			      else:
			       continue
			    else:
			     continue
			   else:
			    continue
			  else:
			   continue
			 else:
			  if image[y+-3][x+0] > cb: 
			   if image[y+-3][x+-1] > cb: 
			    if image[y+-2][x+-2] > cb: 
			     if image[y+-1][x+-3] > cb: 
			      if image[y+0][x+-3] > cb: 
			       if image[y+1][x+-3] > cb: 
			        if image[y+2][x+-2] > cb: 
			         if image[y+3][x+-1] > cb: 
			          pass
			         else:
			          if image[y+-2][x+2] > cb: 
			           if image[y+-3][x+1] > cb: 
			            pass
			           else:
			            continue
			          else:
			           continue
			        else:
			         if image[y+-1][x+3] > cb: 
			          if image[y+-2][x+2] > cb: 
			           if image[y+-3][x+1] > cb: 
			            pass
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			       else:
			        if image[y+0][x+3] > cb: 
			         if image[y+-1][x+3] > cb: 
			          if image[y+-2][x+2] > cb: 
			           if image[y+-3][x+1] > cb: 
			            pass
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			      else:
			       if image[y+1][x+3] > cb: 
			        if image[y+0][x+3] > cb: 
			         if image[y+-1][x+3] > cb: 
			          if image[y+-2][x+2] > cb: 
			           if image[y+-3][x+1] > cb: 
			            pass
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			       else:
			        continue
			     else:
			      if image[y+2][x+2] > cb: 
			       if image[y+1][x+3] > cb: 
			        if image[y+0][x+3] > cb: 
			         if image[y+-1][x+3] > cb: 
			          if image[y+-2][x+2] > cb: 
			           if image[y+-3][x+1] > cb: 
			            pass
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			       else:
			        continue
			      else:
			       continue
			    else:
			     continue
			   else:
			    continue
			  elif image[y+-3][x+0] < c_b: 
			   if image[y+-3][x+1] < c_b:
			    if image[y+-3][x+-1] < c_b:
			     if image[y+-2][x+-2] < c_b:
			      if image[y+-2][x+2] < c_b:
			       if image[y+-1][x+3] < c_b:
			        if image[y+0][x+3] < c_b:
			         if image[y+1][x+3] < c_b:
			          if image[y+2][x+2] < c_b:
			           pass
			          else:
			           if image[y+-1][x+-3] < c_b:
			            pass
			           else:
			            continue
			         else:
			          if image[y+-1][x+-3] < c_b:
			           if image[y+0][x+-3] < c_b:
			            pass
			           else:
			            continue
			          else:
			           continue
			        else:
			         if image[y+-1][x+-3] < c_b:
			          if image[y+0][x+-3] < c_b:
			           if image[y+1][x+-3] < c_b:
			            pass
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			       else:
			        if image[y+-1][x+-3] < c_b:
			         if image[y+0][x+-3] < c_b:
			          if image[y+1][x+-3] < c_b:
			           if image[y+2][x+-2] < c_b:
			            pass
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			      else:
			       if image[y+-1][x+-3] < c_b:
			        if image[y+0][x+-3] < c_b:
			         if image[y+1][x+-3] < c_b:
			          if image[y+2][x+-2] < c_b:
			           if image[y+3][x+-1] < c_b:
			            pass
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			       else:
			        continue
			     else:
			      continue
			    else:
			     continue
			   else:
			    continue
			  else:
			   continue
			elif image[y+3][x+0] < c_b: 
			 if image[y+3][x+1] > cb: 
			  if image[y+-3][x+0] > cb: 
			   if image[y+-3][x+1] > cb: 
			    if image[y+-3][x+-1] > cb: 
			     if image[y+-2][x+2] > cb: 
			      if image[y+-1][x+3] > cb: 
			       if image[y+0][x+3] > cb: 
			        if image[y+1][x+3] > cb: 
			         if image[y+2][x+2] > cb: 
			          pass
			         else:
			          if image[y+-2][x+-2] > cb: 
			           if image[y+-1][x+-3] > cb: 
			            pass
			           else:
			            continue
			          else:
			           continue
			        else:
			         if image[y+-2][x+-2] > cb: 
			          if image[y+-1][x+-3] > cb: 
			           if image[y+0][x+-3] > cb: 
			            pass
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			       else:
			        if image[y+-2][x+-2] > cb: 
			         if image[y+-1][x+-3] > cb: 
			          if image[y+0][x+-3] > cb: 
			           if image[y+1][x+-3] > cb: 
			            pass
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			      else:
			       if image[y+-2][x+-2] > cb: 
			        if image[y+-1][x+-3] > cb: 
			         if image[y+0][x+-3] > cb: 
			          if image[y+1][x+-3] > cb: 
			           if image[y+2][x+-2] > cb: 
			            pass
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			       else:
			        continue
			     else:
			      if image[y+-2][x+-2] > cb: 
			       if image[y+-1][x+-3] > cb: 
			        if image[y+0][x+-3] > cb: 
			         if image[y+1][x+-3] > cb: 
			          if image[y+2][x+-2] > cb: 
			           if image[y+3][x+-1] > cb: 
			            pass
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			       else:
			        continue
			      else:
			       continue
			    else:
			     continue
			   else:
			    continue
			  elif image[y+-3][x+0] < c_b: 
			   if image[y+-3][x+-1] < c_b:
			    if image[y+-2][x+-2] < c_b:
			     if image[y+-1][x+-3] < c_b:
			      if image[y+0][x+-3] < c_b:
			       if image[y+1][x+-3] < c_b:
			        if image[y+2][x+-2] < c_b:
			         if image[y+3][x+-1] < c_b:
			          pass
			         else:
			          if image[y+-2][x+2] < c_b:
			           if image[y+-3][x+1] < c_b:
			            pass
			           else:
			            continue
			          else:
			           continue
			        else:
			         if image[y+-1][x+3] < c_b:
			          if image[y+-2][x+2] < c_b:
			           if image[y+-3][x+1] < c_b:
			            pass
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			       else:
			        if image[y+0][x+3] < c_b:
			         if image[y+-1][x+3] < c_b:
			          if image[y+-2][x+2] < c_b:
			           if image[y+-3][x+1] < c_b:
			            pass
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			      else:
			       if image[y+1][x+3] < c_b:
			        if image[y+0][x+3] < c_b:
			         if image[y+-1][x+3] < c_b:
			          if image[y+-2][x+2] < c_b:
			           if image[y+-3][x+1] < c_b:
			            pass
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			       else:
			        continue
			     else:
			      if image[y+2][x+2] < c_b:
			       if image[y+1][x+3] < c_b:
			        if image[y+0][x+3] < c_b:
			         if image[y+-1][x+3] < c_b:
			          if image[y+-2][x+2] < c_b:
			           if image[y+-3][x+1] < c_b:
			            pass
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			       else:
			        continue
			      else:
			       continue
			    else:
			     continue
			   else:
			    continue
			  else:
			   continue
			 elif image[y+3][x+1] < c_b: 
			  if image[y+2][x+2] > cb: 
			   if image[y+-3][x+-1] > cb: 
			    if image[y+-3][x+1] > cb: 
			     if image[y+-3][x+0] > cb: 
			      if image[y+-2][x+-2] > cb: 
			       if image[y+-2][x+2] > cb: 
			        if image[y+-1][x+3] > cb: 
			         if image[y+0][x+3] > cb: 
			          if image[y+1][x+3] > cb: 
			           pass
			          else:
			           if image[y+-1][x+-3] > cb: 
			            if image[y+0][x+-3] > cb: 
			             pass
			            else:
			             continue
			           else:
			            continue
			         else:
			          if image[y+-1][x+-3] > cb: 
			           if image[y+0][x+-3] > cb: 
			            if image[y+1][x+-3] > cb: 
			             pass
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			        else:
			         if image[y+-1][x+-3] > cb: 
			          if image[y+0][x+-3] > cb: 
			           if image[y+1][x+-3] > cb: 
			            if image[y+2][x+-2] > cb: 
			             pass
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			       else:
			        if image[y+-1][x+-3] > cb: 
			         if image[y+0][x+-3] > cb: 
			          if image[y+1][x+-3] > cb: 
			           if image[y+2][x+-2] > cb: 
			            if image[y+3][x+-1] > cb: 
			             pass
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			      else:
			       continue
			     else:
			      continue
			    else:
			     continue
			   elif image[y+-3][x+-1] < c_b: 
			    if image[y+-2][x+-2] < c_b:
			     if image[y+-1][x+-3] < c_b:
			      if image[y+0][x+-3] < c_b:
			       if image[y+1][x+-3] < c_b:
			        if image[y+2][x+-2] < c_b:
			         if image[y+3][x+-1] < c_b:
			          pass
			         else:
			          if image[y+-2][x+2] < c_b:
			           if image[y+-3][x+1] < c_b:
			            if image[y+-3][x+0] < c_b:
			             pass
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			        else:
			         if image[y+-1][x+3] < c_b:
			          if image[y+-2][x+2] < c_b:
			           if image[y+-3][x+1] < c_b:
			            if image[y+-3][x+0] < c_b:
			             pass
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			       else:
			        if image[y+0][x+3] < c_b:
			         if image[y+-1][x+3] < c_b:
			          if image[y+-2][x+2] < c_b:
			           if image[y+-3][x+1] < c_b:
			            if image[y+-3][x+0] < c_b:
			             pass
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			      else:
			       if image[y+1][x+3] < c_b:
			        if image[y+0][x+3] < c_b:
			         if image[y+-1][x+3] < c_b:
			          if image[y+-2][x+2] < c_b:
			           if image[y+-3][x+1] < c_b:
			            if image[y+-3][x+0] < c_b:
			             pass
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			       else:
			        continue
			     else:
			      continue
			    else:
			     continue
			   else:
			    continue
			  elif image[y+2][x+2] < c_b: 
			   if image[y+1][x+3] > cb: 
			    if image[y+-2][x+-2] > cb: 
			     if image[y+-3][x+1] > cb: 
			      if image[y+-3][x+0] > cb: 
			       if image[y+-3][x+-1] > cb: 
			        if image[y+-1][x+-3] > cb: 
			         if image[y+-2][x+2] > cb: 
			          if image[y+-1][x+3] > cb: 
			           if image[y+0][x+3] > cb: 
			            pass
			           else:
			            if image[y+0][x+-3] > cb: 
			             if image[y+1][x+-3] > cb: 
			              pass
			             else:
			              continue
			            else:
			             continue
			          else:
			           if image[y+0][x+-3] > cb: 
			            if image[y+1][x+-3] > cb: 
			             if image[y+2][x+-2] > cb: 
			              pass
			             else:
			              continue
			            else:
			             continue
			           else:
			            continue
			         else:
			          if image[y+0][x+-3] > cb: 
			           if image[y+1][x+-3] > cb: 
			            if image[y+2][x+-2] > cb: 
			             if image[y+3][x+-1] > cb: 
			              pass
			             else:
			              continue
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			        else:
			         continue
			       else:
			        continue
			      else:
			       continue
			     else:
			      continue
			    elif image[y+-2][x+-2] < c_b: 
			     if image[y+-1][x+-3] < c_b:
			      if image[y+0][x+-3] < c_b:
			       if image[y+1][x+-3] < c_b:
			        if image[y+2][x+-2] < c_b:
			         if image[y+3][x+-1] < c_b:
			          pass
			         else:
			          if image[y+-2][x+2] < c_b:
			           if image[y+-3][x+1] < c_b:
			            if image[y+-3][x+0] < c_b:
			             if image[y+-3][x+-1] < c_b:
			              pass
			             else:
			              continue
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			        else:
			         if image[y+-1][x+3] < c_b:
			          if image[y+-2][x+2] < c_b:
			           if image[y+-3][x+1] < c_b:
			            if image[y+-3][x+0] < c_b:
			             if image[y+-3][x+-1] < c_b:
			              pass
			             else:
			              continue
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			       else:
			        if image[y+0][x+3] < c_b:
			         if image[y+-1][x+3] < c_b:
			          if image[y+-2][x+2] < c_b:
			           if image[y+-3][x+1] < c_b:
			            if image[y+-3][x+0] < c_b:
			             if image[y+-3][x+-1] < c_b:
			              pass
			             else:
			              continue
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			      else:
			       continue
			     else:
			      continue
			    else:
			     continue
			   elif image[y+1][x+3] < c_b: 
			    if image[y+0][x+3] > cb: 
			     if image[y+1][x+-3] > cb: 
			      if image[y+-3][x+1] > cb: 
			       if image[y+-3][x+0] > cb: 
			        if image[y+-3][x+-1] > cb: 
			         if image[y+-2][x+-2] > cb: 
			          if image[y+-1][x+-3] > cb: 
			           if image[y+0][x+-3] > cb: 
			            if image[y+-2][x+2] > cb: 
			             if image[y+-1][x+3] > cb: 
			              pass
			             else:
			              if image[y+2][x+-2] > cb: 
			               pass
			              else:
			               continue
			            else:
			             if image[y+2][x+-2] > cb: 
			              if image[y+3][x+-1] > cb: 
			               pass
			              else:
			               continue
			             else:
			              continue
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			       else:
			        continue
			      else:
			       continue
			     elif image[y+1][x+-3] < c_b: 
			      if image[y+-1][x+-3] > cb: 
			       if image[y+-1][x+3] > cb: 
			        if image[y+-2][x+2] > cb: 
			         if image[y+-3][x+1] > cb: 
			          if image[y+-3][x+0] > cb: 
			           if image[y+-3][x+-1] > cb: 
			            if image[y+-2][x+-2] > cb: 
			             if image[y+0][x+-3] > cb: 
			              pass
			             else:
			              continue
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			       else:
			        continue
			      elif image[y+-1][x+-3] < c_b: 
			       if image[y+0][x+-3] < c_b:
			        if image[y+2][x+-2] < c_b:
			         if image[y+3][x+-1] < c_b:
			          pass
			         else:
			          if image[y+-2][x+2] < c_b:
			           if image[y+-3][x+1] < c_b:
			            if image[y+-3][x+0] < c_b:
			             if image[y+-3][x+-1] < c_b:
			              if image[y+-2][x+-2] < c_b:
			               pass
			              else:
			               continue
			             else:
			              continue
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			        else:
			         if image[y+-1][x+3] < c_b:
			          if image[y+-2][x+2] < c_b:
			           if image[y+-3][x+1] < c_b:
			            if image[y+-3][x+0] < c_b:
			             if image[y+-3][x+-1] < c_b:
			              if image[y+-2][x+-2] < c_b:
			               pass
			              else:
			               continue
			             else:
			              continue
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			       else:
			        continue
			      else:
			       continue
			     else:
			      if image[y+-1][x+3] > cb: 
			       if image[y+-2][x+2] > cb: 
			        if image[y+-3][x+1] > cb: 
			         if image[y+-3][x+0] > cb: 
			          if image[y+-3][x+-1] > cb: 
			           if image[y+-2][x+-2] > cb: 
			            if image[y+-1][x+-3] > cb: 
			             if image[y+0][x+-3] > cb: 
			              pass
			             else:
			              continue
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			       else:
			        continue
			      else:
			       continue
			    elif image[y+0][x+3] < c_b: 
			     if image[y+-1][x+3] > cb: 
			      if image[y+2][x+-2] > cb: 
			       if image[y+-3][x+1] > cb: 
			        if image[y+-3][x+0] > cb: 
			         if image[y+-3][x+-1] > cb: 
			          if image[y+-2][x+-2] > cb: 
			           if image[y+-1][x+-3] > cb: 
			            if image[y+0][x+-3] > cb: 
			             if image[y+1][x+-3] > cb: 
			              if image[y+-2][x+2] > cb: 
			               pass
			              else:
			               if image[y+3][x+-1] > cb: 
			                pass
			               else:
			                continue
			             else:
			              continue
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			       else:
			        continue
			      elif image[y+2][x+-2] < c_b: 
			       if image[y+0][x+-3] > cb: 
			        if image[y+-2][x+2] > cb: 
			         if image[y+-3][x+1] > cb: 
			          if image[y+-3][x+0] > cb: 
			           if image[y+-3][x+-1] > cb: 
			            if image[y+-2][x+-2] > cb: 
			             if image[y+-1][x+-3] > cb: 
			              if image[y+1][x+-3] > cb: 
			               pass
			              else:
			               continue
			             else:
			              continue
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			       elif image[y+0][x+-3] < c_b: 
			        if image[y+1][x+-3] < c_b:
			         if image[y+3][x+-1] < c_b:
			          pass
			         else:
			          if image[y+-2][x+2] < c_b:
			           if image[y+-3][x+1] < c_b:
			            if image[y+-3][x+0] < c_b:
			             if image[y+-3][x+-1] < c_b:
			              if image[y+-2][x+-2] < c_b:
			               if image[y+-1][x+-3] < c_b:
			                pass
			               else:
			                continue
			              else:
			               continue
			             else:
			              continue
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			        else:
			         continue
			       else:
			        continue
			      else:
			       if image[y+-2][x+2] > cb: 
			        if image[y+-3][x+1] > cb: 
			         if image[y+-3][x+0] > cb: 
			          if image[y+-3][x+-1] > cb: 
			           if image[y+-2][x+-2] > cb: 
			            if image[y+-1][x+-3] > cb: 
			             if image[y+0][x+-3] > cb: 
			              if image[y+1][x+-3] > cb: 
			               pass
			              else:
			               continue
			             else:
			              continue
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			       else:
			        continue
			     elif image[y+-1][x+3] < c_b: 
			      if image[y+-2][x+2] > cb: 
			       if image[y+3][x+-1] < c_b:
			        if image[y+1][x+-3] > cb: 
			         if image[y+-3][x+1] > cb: 
			          if image[y+-3][x+0] > cb: 
			           if image[y+-3][x+-1] > cb: 
			            if image[y+-2][x+-2] > cb: 
			             if image[y+-1][x+-3] > cb: 
			              if image[y+0][x+-3] > cb: 
			               if image[y+2][x+-2] > cb: 
			                pass
			               else:
			                continue
			              else:
			               continue
			             else:
			              continue
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			        elif image[y+1][x+-3] < c_b: 
			         if image[y+2][x+-2] < c_b:
			          pass
			         else:
			          continue
			        else:
			         continue
			       else:
			        if image[y+-3][x+1] > cb: 
			         if image[y+-3][x+0] > cb: 
			          if image[y+-3][x+-1] > cb: 
			           if image[y+-2][x+-2] > cb: 
			            if image[y+-1][x+-3] > cb: 
			             if image[y+0][x+-3] > cb: 
			              if image[y+1][x+-3] > cb: 
			               if image[y+2][x+-2] > cb: 
			                pass
			               else:
			                continue
			              else:
			               continue
			             else:
			              continue
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			      elif image[y+-2][x+2] < c_b: 
			       if image[y+-3][x+1] > cb: 
			        if image[y+2][x+-2] > cb: 
			         if image[y+-3][x+0] > cb: 
			          if image[y+-3][x+-1] > cb: 
			           if image[y+-2][x+-2] > cb: 
			            if image[y+-1][x+-3] > cb: 
			             if image[y+0][x+-3] > cb: 
			              if image[y+1][x+-3] > cb: 
			               if image[y+3][x+-1] > cb: 
			                pass
			               else:
			                continue
			              else:
			               continue
			             else:
			              continue
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			        elif image[y+2][x+-2] < c_b: 
			         if image[y+3][x+-1] < c_b:
			          pass
			         else:
			          continue
			        else:
			         continue
			       elif image[y+-3][x+1] < c_b: 
			        if image[y+-3][x+0] < c_b:
			         pass
			        else:
			         if image[y+3][x+-1] < c_b:
			          pass
			         else:
			          continue
			       else:
			        if image[y+2][x+-2] < c_b:
			         if image[y+3][x+-1] < c_b:
			          pass
			         else:
			          continue
			        else:
			         continue
			      else:
			       if image[y+1][x+-3] > cb: 
			        if image[y+-3][x+1] > cb: 
			         if image[y+-3][x+0] > cb: 
			          if image[y+-3][x+-1] > cb: 
			           if image[y+-2][x+-2] > cb: 
			            if image[y+-1][x+-3] > cb: 
			             if image[y+0][x+-3] > cb: 
			              if image[y+2][x+-2] > cb: 
			               if image[y+3][x+-1] > cb: 
			                pass
			               else:
			                continue
			              else:
			               continue
			             else:
			              continue
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			       elif image[y+1][x+-3] < c_b: 
			        if image[y+2][x+-2] < c_b:
			         if image[y+3][x+-1] < c_b:
			          pass
			         else:
			          continue
			        else:
			         continue
			       else:
			        continue
			     else:
			      if image[y+0][x+-3] > cb: 
			       if image[y+-3][x+1] > cb: 
			        if image[y+-3][x+0] > cb: 
			         if image[y+-3][x+-1] > cb: 
			          if image[y+-2][x+-2] > cb: 
			           if image[y+-1][x+-3] > cb: 
			            if image[y+1][x+-3] > cb: 
			             if image[y+2][x+-2] > cb: 
			              if image[y+-2][x+2] > cb: 
			               pass
			              else:
			               if image[y+3][x+-1] > cb: 
			                pass
			               else:
			                continue
			             else:
			              continue
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			       else:
			        continue
			      elif image[y+0][x+-3] < c_b: 
			       if image[y+1][x+-3] < c_b:
			        if image[y+2][x+-2] < c_b:
			         if image[y+3][x+-1] < c_b:
			          pass
			         else:
			          if image[y+-2][x+2] < c_b:
			           if image[y+-3][x+1] < c_b:
			            if image[y+-3][x+0] < c_b:
			             if image[y+-3][x+-1] < c_b:
			              if image[y+-2][x+-2] < c_b:
			               if image[y+-1][x+-3] < c_b:
			                pass
			               else:
			                continue
			              else:
			               continue
			             else:
			              continue
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			        else:
			         continue
			       else:
			        continue
			      else:
			       continue
			    else:
			     if image[y+-1][x+-3] > cb: 
			      if image[y+-3][x+1] > cb: 
			       if image[y+-3][x+0] > cb: 
			        if image[y+-3][x+-1] > cb: 
			         if image[y+-2][x+-2] > cb: 
			          if image[y+0][x+-3] > cb: 
			           if image[y+1][x+-3] > cb: 
			            if image[y+-2][x+2] > cb: 
			             if image[y+-1][x+3] > cb: 
			              pass
			             else:
			              if image[y+2][x+-2] > cb: 
			               pass
			              else:
			               continue
			            else:
			             if image[y+2][x+-2] > cb: 
			              if image[y+3][x+-1] > cb: 
			               pass
			              else:
			               continue
			             else:
			              continue
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			       else:
			        continue
			      else:
			       continue
			     elif image[y+-1][x+-3] < c_b: 
			      if image[y+0][x+-3] < c_b:
			       if image[y+1][x+-3] < c_b:
			        if image[y+2][x+-2] < c_b:
			         if image[y+3][x+-1] < c_b:
			          pass
			         else:
			          if image[y+-2][x+2] < c_b:
			           if image[y+-3][x+1] < c_b:
			            if image[y+-3][x+0] < c_b:
			             if image[y+-3][x+-1] < c_b:
			              if image[y+-2][x+-2] < c_b:
			               pass
			              else:
			               continue
			             else:
			              continue
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			        else:
			         if image[y+-1][x+3] < c_b:
			          if image[y+-2][x+2] < c_b:
			           if image[y+-3][x+1] < c_b:
			            if image[y+-3][x+0] < c_b:
			             if image[y+-3][x+-1] < c_b:
			              if image[y+-2][x+-2] < c_b:
			               pass
			              else:
			               continue
			             else:
			              continue
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			       else:
			        continue
			      else:
			       continue
			     else:
			      continue
			   else:
			    if image[y+-2][x+-2] > cb: 
			     if image[y+-3][x+1] > cb: 
			      if image[y+-3][x+0] > cb: 
			       if image[y+-3][x+-1] > cb: 
			        if image[y+-1][x+-3] > cb: 
			         if image[y+0][x+-3] > cb: 
			          if image[y+-2][x+2] > cb: 
			           if image[y+-1][x+3] > cb: 
			            if image[y+0][x+3] > cb: 
			             pass
			            else:
			             if image[y+1][x+-3] > cb: 
			              pass
			             else:
			              continue
			           else:
			            if image[y+1][x+-3] > cb: 
			             if image[y+2][x+-2] > cb: 
			              pass
			             else:
			              continue
			            else:
			             continue
			          else:
			           if image[y+1][x+-3] > cb: 
			            if image[y+2][x+-2] > cb: 
			             if image[y+3][x+-1] > cb: 
			              pass
			             else:
			              continue
			            else:
			             continue
			           else:
			            continue
			         else:
			          continue
			        else:
			         continue
			       else:
			        continue
			      else:
			       continue
			     else:
			      continue
			    elif image[y+-2][x+-2] < c_b: 
			     if image[y+-1][x+-3] < c_b:
			      if image[y+0][x+-3] < c_b:
			       if image[y+1][x+-3] < c_b:
			        if image[y+2][x+-2] < c_b:
			         if image[y+3][x+-1] < c_b:
			          pass
			         else:
			          if image[y+-2][x+2] < c_b:
			           if image[y+-3][x+1] < c_b:
			            if image[y+-3][x+0] < c_b:
			             if image[y+-3][x+-1] < c_b:
			              pass
			             else:
			              continue
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			        else:
			         if image[y+-1][x+3] < c_b:
			          if image[y+-2][x+2] < c_b:
			           if image[y+-3][x+1] < c_b:
			            if image[y+-3][x+0] < c_b:
			             if image[y+-3][x+-1] < c_b:
			              pass
			             else:
			              continue
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			       else:
			        if image[y+0][x+3] < c_b:
			         if image[y+-1][x+3] < c_b:
			          if image[y+-2][x+2] < c_b:
			           if image[y+-3][x+1] < c_b:
			            if image[y+-3][x+0] < c_b:
			             if image[y+-3][x+-1] < c_b:
			              pass
			             else:
			              continue
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			      else:
			       continue
			     else:
			      continue
			    else:
			     continue
			  else:
			   if image[y+-3][x+-1] > cb: 
			    if image[y+-3][x+1] > cb: 
			     if image[y+-3][x+0] > cb: 
			      if image[y+-2][x+-2] > cb: 
			       if image[y+-1][x+-3] > cb: 
			        if image[y+-2][x+2] > cb: 
			         if image[y+-1][x+3] > cb: 
			          if image[y+0][x+3] > cb: 
			           if image[y+1][x+3] > cb: 
			            pass
			           else:
			            if image[y+0][x+-3] > cb: 
			             pass
			            else:
			             continue
			          else:
			           if image[y+0][x+-3] > cb: 
			            if image[y+1][x+-3] > cb: 
			             pass
			            else:
			             continue
			           else:
			            continue
			         else:
			          if image[y+0][x+-3] > cb: 
			           if image[y+1][x+-3] > cb: 
			            if image[y+2][x+-2] > cb: 
			             pass
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			        else:
			         if image[y+0][x+-3] > cb: 
			          if image[y+1][x+-3] > cb: 
			           if image[y+2][x+-2] > cb: 
			            if image[y+3][x+-1] > cb: 
			             pass
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			       else:
			        continue
			      else:
			       continue
			     else:
			      continue
			    else:
			     continue
			   elif image[y+-3][x+-1] < c_b: 
			    if image[y+-2][x+-2] < c_b:
			     if image[y+-1][x+-3] < c_b:
			      if image[y+0][x+-3] < c_b:
			       if image[y+1][x+-3] < c_b:
			        if image[y+2][x+-2] < c_b:
			         if image[y+3][x+-1] < c_b:
			          pass
			         else:
			          if image[y+-2][x+2] < c_b:
			           if image[y+-3][x+1] < c_b:
			            if image[y+-3][x+0] < c_b:
			             pass
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			        else:
			         if image[y+-1][x+3] < c_b:
			          if image[y+-2][x+2] < c_b:
			           if image[y+-3][x+1] < c_b:
			            if image[y+-3][x+0] < c_b:
			             pass
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			       else:
			        if image[y+0][x+3] < c_b:
			         if image[y+-1][x+3] < c_b:
			          if image[y+-2][x+2] < c_b:
			           if image[y+-3][x+1] < c_b:
			            if image[y+-3][x+0] < c_b:
			             pass
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			      else:
			       if image[y+1][x+3] < c_b:
			        if image[y+0][x+3] < c_b:
			         if image[y+-1][x+3] < c_b:
			          if image[y+-2][x+2] < c_b:
			           if image[y+-3][x+1] < c_b:
			            if image[y+-3][x+0] < c_b:
			             pass
			            else:
			             continue
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			       else:
			        continue
			     else:
			      continue
			    else:
			     continue
			   else:
			    continue
			 else:
			  if image[y+-3][x+0] > cb: 
			   if image[y+-3][x+1] > cb: 
			    if image[y+-3][x+-1] > cb: 
			     if image[y+-2][x+-2] > cb: 
			      if image[y+-2][x+2] > cb: 
			       if image[y+-1][x+3] > cb: 
			        if image[y+0][x+3] > cb: 
			         if image[y+1][x+3] > cb: 
			          if image[y+2][x+2] > cb: 
			           pass
			          else:
			           if image[y+-1][x+-3] > cb: 
			            pass
			           else:
			            continue
			         else:
			          if image[y+-1][x+-3] > cb: 
			           if image[y+0][x+-3] > cb: 
			            pass
			           else:
			            continue
			          else:
			           continue
			        else:
			         if image[y+-1][x+-3] > cb: 
			          if image[y+0][x+-3] > cb: 
			           if image[y+1][x+-3] > cb: 
			            pass
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			       else:
			        if image[y+-1][x+-3] > cb: 
			         if image[y+0][x+-3] > cb: 
			          if image[y+1][x+-3] > cb: 
			           if image[y+2][x+-2] > cb: 
			            pass
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			      else:
			       if image[y+-1][x+-3] > cb: 
			        if image[y+0][x+-3] > cb: 
			         if image[y+1][x+-3] > cb: 
			          if image[y+2][x+-2] > cb: 
			           if image[y+3][x+-1] > cb: 
			            pass
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			       else:
			        continue
			     else:
			      continue
			    else:
			     continue
			   else:
			    continue
			  elif image[y+-3][x+0] < c_b: 
			   if image[y+-3][x+-1] < c_b:
			    if image[y+-2][x+-2] < c_b:
			     if image[y+-1][x+-3] < c_b:
			      if image[y+0][x+-3] < c_b:
			       if image[y+1][x+-3] < c_b:
			        if image[y+2][x+-2] < c_b:
			         if image[y+3][x+-1] < c_b:
			          pass
			         else:
			          if image[y+-2][x+2] < c_b:
			           if image[y+-3][x+1] < c_b:
			            pass
			           else:
			            continue
			          else:
			           continue
			        else:
			         if image[y+-1][x+3] < c_b:
			          if image[y+-2][x+2] < c_b:
			           if image[y+-3][x+1] < c_b:
			            pass
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			       else:
			        if image[y+0][x+3] < c_b:
			         if image[y+-1][x+3] < c_b:
			          if image[y+-2][x+2] < c_b:
			           if image[y+-3][x+1] < c_b:
			            pass
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			      else:
			       if image[y+1][x+3] < c_b:
			        if image[y+0][x+3] < c_b:
			         if image[y+-1][x+3] < c_b:
			          if image[y+-2][x+2] < c_b:
			           if image[y+-3][x+1] < c_b:
			            pass
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			       else:
			        continue
			     else:
			      if image[y+2][x+2] < c_b:
			       if image[y+1][x+3] < c_b:
			        if image[y+0][x+3] < c_b:
			         if image[y+-1][x+3] < c_b:
			          if image[y+-2][x+2] < c_b:
			           if image[y+-3][x+1] < c_b:
			            pass
			           else:
			            continue
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			       else:
			        continue
			      else:
			       continue
			    else:
			     continue
			   else:
			    continue
			  else:
			   continue
			else:
			 if image[y+-3][x+1] > cb: 
			  if image[y+-3][x+0] > cb: 
			   if image[y+-3][x+-1] > cb: 
			    if image[y+-2][x+2] > cb: 
			     if image[y+-1][x+3] > cb: 
			      if image[y+0][x+3] > cb: 
			       if image[y+1][x+3] > cb: 
			        if image[y+2][x+2] > cb: 
			         if image[y+3][x+1] > cb: 
			          pass
			         else:
			          if image[y+-2][x+-2] > cb: 
			           pass
			          else:
			           continue
			        else:
			         if image[y+-2][x+-2] > cb: 
			          if image[y+-1][x+-3] > cb: 
			           pass
			          else:
			           continue
			         else:
			          continue
			       else:
			        if image[y+-2][x+-2] > cb: 
			         if image[y+-1][x+-3] > cb: 
			          if image[y+0][x+-3] > cb: 
			           pass
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			      else:
			       if image[y+-2][x+-2] > cb: 
			        if image[y+-1][x+-3] > cb: 
			         if image[y+0][x+-3] > cb: 
			          if image[y+1][x+-3] > cb: 
			           pass
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			       else:
			        continue
			     else:
			      if image[y+-2][x+-2] > cb: 
			       if image[y+-1][x+-3] > cb: 
			        if image[y+0][x+-3] > cb: 
			         if image[y+1][x+-3] > cb: 
			          if image[y+2][x+-2] > cb: 
			           pass
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			       else:
			        continue
			      else:
			       continue
			    else:
			     if image[y+-2][x+-2] > cb: 
			      if image[y+-1][x+-3] > cb: 
			       if image[y+0][x+-3] > cb: 
			        if image[y+1][x+-3] > cb: 
			         if image[y+2][x+-2] > cb: 
			          if image[y+3][x+-1] > cb: 
			           pass
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			       else:
			        continue
			      else:
			       continue
			     else:
			      continue
			   else:
			    continue
			  else:
			   continue
			 elif image[y+-3][x+1] < c_b: 
			  if image[y+-3][x+0] < c_b:
			   if image[y+-3][x+-1] < c_b:
			    if image[y+-2][x+2] < c_b:
			     if image[y+-1][x+3] < c_b:
			      if image[y+0][x+3] < c_b:
			       if image[y+1][x+3] < c_b:
			        if image[y+2][x+2] < c_b:
			         if image[y+3][x+1] < c_b:
			          pass
			         else:
			          if image[y+-2][x+-2] < c_b:
			           pass
			          else:
			           continue
			        else:
			         if image[y+-2][x+-2] < c_b:
			          if image[y+-1][x+-3] < c_b:
			           pass
			          else:
			           continue
			         else:
			          continue
			       else:
			        if image[y+-2][x+-2] < c_b:
			         if image[y+-1][x+-3] < c_b:
			          if image[y+0][x+-3] < c_b:
			           pass
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			      else:
			       if image[y+-2][x+-2] < c_b:
			        if image[y+-1][x+-3] < c_b:
			         if image[y+0][x+-3] < c_b:
			          if image[y+1][x+-3] < c_b:
			           pass
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			       else:
			        continue
			     else:
			      if image[y+-2][x+-2] < c_b:
			       if image[y+-1][x+-3] < c_b:
			        if image[y+0][x+-3] < c_b:
			         if image[y+1][x+-3] < c_b:
			          if image[y+2][x+-2] < c_b:
			           pass
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			       else:
			        continue
			      else:
			       continue
			    else:
			     if image[y+-2][x+-2] < c_b:
			      if image[y+-1][x+-3] < c_b:
			       if image[y+0][x+-3] < c_b:
			        if image[y+1][x+-3] < c_b:
			         if image[y+2][x+-2] < c_b:
			          if image[y+3][x+-1] < c_b:
			           pass
			          else:
			           continue
			         else:
			          continue
			        else:
			         continue
			       else:
			        continue
			      else:
			       continue
			     else:
			      continue
			   else:
			    continue
			  else:
			   continue
			 else:
			  continue


			corners.append((x,y))

	for i in xrange(0, len(corners)):
		scores.append(corner_score(image, corners[i][0], corners[i][1]))
	
	if do_nonmax:
		#Paint corners into an image
		sc = numpy.zeros(image.shape);
		for i in xrange(0, len(corners)):
			sc[corners[i][1], corners[i][0]] = scores[i];

		nonmax_corners=[]
		nonmax_scores =[]

		for i in xrange(0, len(corners)):
			s = scores[i]
			x = corners[i][0]
			y = corners[i][1]

			if s >= sc[y-1][x+1] and s >= sc[y-1][x] and s >= sc[y-1][x-1] and s >= sc[y][x+1] and s >= sc[y][x-1] and s >= sc[y+1][x+1] and s >= sc[y+1][x] and s >= sc[y+1][x-1]:
				nonmax_corners.append((x, y))
				nonmax_scores.append(s)


		return (nonmax_corners, nonmax_scores)
			


	else:
		return (corners, scores)


def is_a_corner(i, posx, posy, b):
	cb = i[posy][posx] + b
	c_b = i[posy][posx] - b
	if i[posy+3][posx+0] > cb: 
	 if i[posy+3][posx+1] > cb: 
	  if i[posy+2][posx+2] > cb: 
	   if i[posy+1][posx+3] > cb: 
	    if i[posy+0][posx+3] > cb: 
	     if i[posy+-1][posx+3] > cb: 
	      if i[posy+-2][posx+2] > cb: 
	       if i[posy+-3][posx+1] > cb: 
	        if i[posy+-3][posx+0] > cb: 
	         return 1
	        else:
	         if i[posy+3][posx+-1] > cb: 
	          return 1
	         else:
	          return 0
	       elif i[posy+-3][posx+1] < c_b: 
	        if i[posy+2][posx+-2] > cb: 
	         if i[posy+3][posx+-1] > cb: 
	          return 1
	         else:
	          return 0
	        elif i[posy+2][posx+-2] < c_b: 
	         if i[posy+-3][posx+0] < c_b:
	          if i[posy+-3][posx+-1] < c_b:
	           if i[posy+-2][posx+-2] < c_b:
	            if i[posy+-1][posx+-3] < c_b:
	             if i[posy+0][posx+-3] < c_b:
	              if i[posy+1][posx+-3] < c_b:
	               if i[posy+3][posx+-1] < c_b:
	                return 1
	               else:
	                return 0
	              else:
	               return 0
	             else:
	              return 0
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	       else:
	        if i[posy+2][posx+-2] > cb: 
	         if i[posy+3][posx+-1] > cb: 
	          return 1
	         else:
	          return 0
	        else:
	         return 0
	      elif i[posy+-2][posx+2] < c_b: 
	       if i[posy+3][posx+-1] > cb: 
	        if i[posy+1][posx+-3] > cb: 
	         if i[posy+2][posx+-2] > cb: 
	          return 1
	         else:
	          return 0
	        elif i[posy+1][posx+-3] < c_b: 
	         if i[posy+-3][posx+1] < c_b:
	          if i[posy+-3][posx+0] < c_b:
	           if i[posy+-3][posx+-1] < c_b:
	            if i[posy+-2][posx+-2] < c_b:
	             if i[posy+-1][posx+-3] < c_b:
	              if i[posy+0][posx+-3] < c_b:
	               if i[posy+2][posx+-2] < c_b:
	                return 1
	               else:
	                return 0
	              else:
	               return 0
	             else:
	              return 0
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	       else:
	        if i[posy+-3][posx+1] < c_b:
	         if i[posy+-3][posx+0] < c_b:
	          if i[posy+-3][posx+-1] < c_b:
	           if i[posy+-2][posx+-2] < c_b:
	            if i[posy+-1][posx+-3] < c_b:
	             if i[posy+0][posx+-3] < c_b:
	              if i[posy+1][posx+-3] < c_b:
	               if i[posy+2][posx+-2] < c_b:
	                return 1
	               else:
	                return 0
	              else:
	               return 0
	             else:
	              return 0
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	      else:
	       if i[posy+1][posx+-3] > cb: 
	        if i[posy+2][posx+-2] > cb: 
	         if i[posy+3][posx+-1] > cb: 
	          return 1
	         else:
	          return 0
	        else:
	         return 0
	       elif i[posy+1][posx+-3] < c_b: 
	        if i[posy+-3][posx+1] < c_b:
	         if i[posy+-3][posx+0] < c_b:
	          if i[posy+-3][posx+-1] < c_b:
	           if i[posy+-2][posx+-2] < c_b:
	            if i[posy+-1][posx+-3] < c_b:
	             if i[posy+0][posx+-3] < c_b:
	              if i[posy+2][posx+-2] < c_b:
	               if i[posy+3][posx+-1] < c_b:
	                return 1
	               else:
	                return 0
	              else:
	               return 0
	             else:
	              return 0
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	       else:
	        return 0
	     elif i[posy+-1][posx+3] < c_b: 
	      if i[posy+2][posx+-2] > cb: 
	       if i[posy+0][posx+-3] > cb: 
	        if i[posy+1][posx+-3] > cb: 
	         if i[posy+3][posx+-1] > cb: 
	          return 1
	         else:
	          if i[posy+-2][posx+2] > cb: 
	           if i[posy+-3][posx+1] > cb: 
	            if i[posy+-3][posx+0] > cb: 
	             if i[posy+-3][posx+-1] > cb: 
	              if i[posy+-2][posx+-2] > cb: 
	               if i[posy+-1][posx+-3] > cb: 
	                return 1
	               else:
	                return 0
	              else:
	               return 0
	             else:
	              return 0
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	        else:
	         return 0
	       elif i[posy+0][posx+-3] < c_b: 
	        if i[posy+-2][posx+2] < c_b:
	         if i[posy+-3][posx+1] < c_b:
	          if i[posy+-3][posx+0] < c_b:
	           if i[posy+-3][posx+-1] < c_b:
	            if i[posy+-2][posx+-2] < c_b:
	             if i[posy+-1][posx+-3] < c_b:
	              if i[posy+1][posx+-3] < c_b:
	               return 1
	              else:
	               return 0
	             else:
	              return 0
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	       else:
	        return 0
	      elif i[posy+2][posx+-2] < c_b: 
	       if i[posy+-3][posx+1] < c_b:
	        if i[posy+-3][posx+0] < c_b:
	         if i[posy+-3][posx+-1] < c_b:
	          if i[posy+-2][posx+-2] < c_b:
	           if i[posy+-1][posx+-3] < c_b:
	            if i[posy+0][posx+-3] < c_b:
	             if i[posy+1][posx+-3] < c_b:
	              if i[posy+-2][posx+2] < c_b:
	               return 1
	              else:
	               if i[posy+3][posx+-1] < c_b:
	                return 1
	               else:
	                return 0
	             else:
	              return 0
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	       else:
	        return 0
	      else:
	       if i[posy+-2][posx+2] < c_b:
	        if i[posy+-3][posx+1] < c_b:
	         if i[posy+-3][posx+0] < c_b:
	          if i[posy+-3][posx+-1] < c_b:
	           if i[posy+-2][posx+-2] < c_b:
	            if i[posy+-1][posx+-3] < c_b:
	             if i[posy+0][posx+-3] < c_b:
	              if i[posy+1][posx+-3] < c_b:
	               return 1
	              else:
	               return 0
	             else:
	              return 0
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	       else:
	        return 0
	     else:
	      if i[posy+0][posx+-3] > cb: 
	       if i[posy+1][posx+-3] > cb: 
	        if i[posy+2][posx+-2] > cb: 
	         if i[posy+3][posx+-1] > cb: 
	          return 1
	         else:
	          if i[posy+-2][posx+2] > cb: 
	           if i[posy+-3][posx+1] > cb: 
	            if i[posy+-3][posx+0] > cb: 
	             if i[posy+-3][posx+-1] > cb: 
	              if i[posy+-2][posx+-2] > cb: 
	               if i[posy+-1][posx+-3] > cb: 
	                return 1
	               else:
	                return 0
	              else:
	               return 0
	             else:
	              return 0
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	        else:
	         return 0
	       else:
	        return 0
	      elif i[posy+0][posx+-3] < c_b: 
	       if i[posy+-3][posx+1] < c_b:
	        if i[posy+-3][posx+0] < c_b:
	         if i[posy+-3][posx+-1] < c_b:
	          if i[posy+-2][posx+-2] < c_b:
	           if i[posy+-1][posx+-3] < c_b:
	            if i[posy+1][posx+-3] < c_b:
	             if i[posy+2][posx+-2] < c_b:
	              if i[posy+-2][posx+2] < c_b:
	               return 1
	              else:
	               if i[posy+3][posx+-1] < c_b:
	                return 1
	               else:
	                return 0
	             else:
	              return 0
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	       else:
	        return 0
	      else:
	       return 0
	    elif i[posy+0][posx+3] < c_b: 
	     if i[posy+1][posx+-3] > cb: 
	      if i[posy+-1][posx+-3] > cb: 
	       if i[posy+0][posx+-3] > cb: 
	        if i[posy+2][posx+-2] > cb: 
	         if i[posy+3][posx+-1] > cb: 
	          return 1
	         else:
	          if i[posy+-2][posx+2] > cb: 
	           if i[posy+-3][posx+1] > cb: 
	            if i[posy+-3][posx+0] > cb: 
	             if i[posy+-3][posx+-1] > cb: 
	              if i[posy+-2][posx+-2] > cb: 
	               return 1
	              else:
	               return 0
	             else:
	              return 0
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	        else:
	         if i[posy+-1][posx+3] > cb: 
	          if i[posy+-2][posx+2] > cb: 
	           if i[posy+-3][posx+1] > cb: 
	            if i[posy+-3][posx+0] > cb: 
	             if i[posy+-3][posx+-1] > cb: 
	              if i[posy+-2][posx+-2] > cb: 
	               return 1
	              else:
	               return 0
	             else:
	              return 0
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	       else:
	        return 0
	      elif i[posy+-1][posx+-3] < c_b: 
	       if i[posy+-1][posx+3] < c_b:
	        if i[posy+-2][posx+2] < c_b:
	         if i[posy+-3][posx+1] < c_b:
	          if i[posy+-3][posx+0] < c_b:
	           if i[posy+-3][posx+-1] < c_b:
	            if i[posy+-2][posx+-2] < c_b:
	             if i[posy+0][posx+-3] < c_b:
	              return 1
	             else:
	              return 0
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	       else:
	        return 0
	      else:
	       return 0
	     elif i[posy+1][posx+-3] < c_b: 
	      if i[posy+-3][posx+1] < c_b:
	       if i[posy+-3][posx+0] < c_b:
	        if i[posy+-3][posx+-1] < c_b:
	         if i[posy+-2][posx+-2] < c_b:
	          if i[posy+-1][posx+-3] < c_b:
	           if i[posy+0][posx+-3] < c_b:
	            if i[posy+-2][posx+2] < c_b:
	             if i[posy+-1][posx+3] < c_b:
	              return 1
	             else:
	              if i[posy+2][posx+-2] < c_b:
	               return 1
	              else:
	               return 0
	            else:
	             if i[posy+2][posx+-2] < c_b:
	              if i[posy+3][posx+-1] < c_b:
	               return 1
	              else:
	               return 0
	             else:
	              return 0
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	       else:
	        return 0
	      else:
	       return 0
	     else:
	      if i[posy+-1][posx+3] < c_b:
	       if i[posy+-2][posx+2] < c_b:
	        if i[posy+-3][posx+1] < c_b:
	         if i[posy+-3][posx+0] < c_b:
	          if i[posy+-3][posx+-1] < c_b:
	           if i[posy+-2][posx+-2] < c_b:
	            if i[posy+-1][posx+-3] < c_b:
	             if i[posy+0][posx+-3] < c_b:
	              return 1
	             else:
	              return 0
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	       else:
	        return 0
	      else:
	       return 0
	    else:
	     if i[posy+-1][posx+-3] > cb: 
	      if i[posy+0][posx+-3] > cb: 
	       if i[posy+1][posx+-3] > cb: 
	        if i[posy+2][posx+-2] > cb: 
	         if i[posy+3][posx+-1] > cb: 
	          return 1
	         else:
	          if i[posy+-2][posx+2] > cb: 
	           if i[posy+-3][posx+1] > cb: 
	            if i[posy+-3][posx+0] > cb: 
	             if i[posy+-3][posx+-1] > cb: 
	              if i[posy+-2][posx+-2] > cb: 
	               return 1
	              else:
	               return 0
	             else:
	              return 0
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	        else:
	         if i[posy+-1][posx+3] > cb: 
	          if i[posy+-2][posx+2] > cb: 
	           if i[posy+-3][posx+1] > cb: 
	            if i[posy+-3][posx+0] > cb: 
	             if i[posy+-3][posx+-1] > cb: 
	              if i[posy+-2][posx+-2] > cb: 
	               return 1
	              else:
	               return 0
	             else:
	              return 0
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	       else:
	        return 0
	      else:
	       return 0
	     elif i[posy+-1][posx+-3] < c_b: 
	      if i[posy+-3][posx+1] < c_b:
	       if i[posy+-3][posx+0] < c_b:
	        if i[posy+-3][posx+-1] < c_b:
	         if i[posy+-2][posx+-2] < c_b:
	          if i[posy+0][posx+-3] < c_b:
	           if i[posy+1][posx+-3] < c_b:
	            if i[posy+-2][posx+2] < c_b:
	             if i[posy+-1][posx+3] < c_b:
	              return 1
	             else:
	              if i[posy+2][posx+-2] < c_b:
	               return 1
	              else:
	               return 0
	            else:
	             if i[posy+2][posx+-2] < c_b:
	              if i[posy+3][posx+-1] < c_b:
	               return 1
	              else:
	               return 0
	             else:
	              return 0
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	       else:
	        return 0
	      else:
	       return 0
	     else:
	      return 0
	   elif i[posy+1][posx+3] < c_b: 
	    if i[posy+-2][posx+-2] > cb: 
	     if i[posy+-1][posx+-3] > cb: 
	      if i[posy+0][posx+-3] > cb: 
	       if i[posy+1][posx+-3] > cb: 
	        if i[posy+2][posx+-2] > cb: 
	         if i[posy+3][posx+-1] > cb: 
	          return 1
	         else:
	          if i[posy+-2][posx+2] > cb: 
	           if i[posy+-3][posx+1] > cb: 
	            if i[posy+-3][posx+0] > cb: 
	             if i[posy+-3][posx+-1] > cb: 
	              return 1
	             else:
	              return 0
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	        else:
	         if i[posy+-1][posx+3] > cb: 
	          if i[posy+-2][posx+2] > cb: 
	           if i[posy+-3][posx+1] > cb: 
	            if i[posy+-3][posx+0] > cb: 
	             if i[posy+-3][posx+-1] > cb: 
	              return 1
	             else:
	              return 0
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	       else:
	        if i[posy+0][posx+3] > cb: 
	         if i[posy+-1][posx+3] > cb: 
	          if i[posy+-2][posx+2] > cb: 
	           if i[posy+-3][posx+1] > cb: 
	            if i[posy+-3][posx+0] > cb: 
	             if i[posy+-3][posx+-1] > cb: 
	              return 1
	             else:
	              return 0
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	      else:
	       return 0
	     else:
	      return 0
	    elif i[posy+-2][posx+-2] < c_b: 
	     if i[posy+-3][posx+1] < c_b:
	      if i[posy+-3][posx+0] < c_b:
	       if i[posy+-3][posx+-1] < c_b:
	        if i[posy+-1][posx+-3] < c_b:
	         if i[posy+-2][posx+2] < c_b:
	          if i[posy+-1][posx+3] < c_b:
	           if i[posy+0][posx+3] < c_b:
	            return 1
	           else:
	            if i[posy+0][posx+-3] < c_b:
	             if i[posy+1][posx+-3] < c_b:
	              return 1
	             else:
	              return 0
	            else:
	             return 0
	          else:
	           if i[posy+0][posx+-3] < c_b:
	            if i[posy+1][posx+-3] < c_b:
	             if i[posy+2][posx+-2] < c_b:
	              return 1
	             else:
	              return 0
	            else:
	             return 0
	           else:
	            return 0
	         else:
	          if i[posy+0][posx+-3] < c_b:
	           if i[posy+1][posx+-3] < c_b:
	            if i[posy+2][posx+-2] < c_b:
	             if i[posy+3][posx+-1] < c_b:
	              return 1
	             else:
	              return 0
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	        else:
	         return 0
	       else:
	        return 0
	      else:
	       return 0
	     else:
	      return 0
	    else:
	     return 0
	   else:
	    if i[posy+-2][posx+-2] > cb: 
	     if i[posy+-1][posx+-3] > cb: 
	      if i[posy+0][posx+-3] > cb: 
	       if i[posy+1][posx+-3] > cb: 
	        if i[posy+2][posx+-2] > cb: 
	         if i[posy+3][posx+-1] > cb: 
	          return 1
	         else:
	          if i[posy+-2][posx+2] > cb: 
	           if i[posy+-3][posx+1] > cb: 
	            if i[posy+-3][posx+0] > cb: 
	             if i[posy+-3][posx+-1] > cb: 
	              return 1
	             else:
	              return 0
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	        else:
	         if i[posy+-1][posx+3] > cb: 
	          if i[posy+-2][posx+2] > cb: 
	           if i[posy+-3][posx+1] > cb: 
	            if i[posy+-3][posx+0] > cb: 
	             if i[posy+-3][posx+-1] > cb: 
	              return 1
	             else:
	              return 0
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	       else:
	        if i[posy+0][posx+3] > cb: 
	         if i[posy+-1][posx+3] > cb: 
	          if i[posy+-2][posx+2] > cb: 
	           if i[posy+-3][posx+1] > cb: 
	            if i[posy+-3][posx+0] > cb: 
	             if i[posy+-3][posx+-1] > cb: 
	              return 1
	             else:
	              return 0
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	      else:
	       return 0
	     else:
	      return 0
	    elif i[posy+-2][posx+-2] < c_b: 
	     if i[posy+-3][posx+1] < c_b:
	      if i[posy+-3][posx+0] < c_b:
	       if i[posy+-3][posx+-1] < c_b:
	        if i[posy+-1][posx+-3] < c_b:
	         if i[posy+0][posx+-3] < c_b:
	          if i[posy+-2][posx+2] < c_b:
	           if i[posy+-1][posx+3] < c_b:
	            if i[posy+0][posx+3] < c_b:
	             return 1
	            else:
	             if i[posy+1][posx+-3] < c_b:
	              return 1
	             else:
	              return 0
	           else:
	            if i[posy+1][posx+-3] < c_b:
	             if i[posy+2][posx+-2] < c_b:
	              return 1
	             else:
	              return 0
	            else:
	             return 0
	          else:
	           if i[posy+1][posx+-3] < c_b:
	            if i[posy+2][posx+-2] < c_b:
	             if i[posy+3][posx+-1] < c_b:
	              return 1
	             else:
	              return 0
	            else:
	             return 0
	           else:
	            return 0
	         else:
	          return 0
	        else:
	         return 0
	       else:
	        return 0
	      else:
	       return 0
	     else:
	      return 0
	    else:
	     return 0
	  elif i[posy+2][posx+2] < c_b: 
	   if i[posy+-3][posx+-1] > cb: 
	    if i[posy+-2][posx+-2] > cb: 
	     if i[posy+-1][posx+-3] > cb: 
	      if i[posy+0][posx+-3] > cb: 
	       if i[posy+1][posx+-3] > cb: 
	        if i[posy+2][posx+-2] > cb: 
	         if i[posy+3][posx+-1] > cb: 
	          return 1
	         else:
	          if i[posy+-2][posx+2] > cb: 
	           if i[posy+-3][posx+1] > cb: 
	            if i[posy+-3][posx+0] > cb: 
	             return 1
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	        else:
	         if i[posy+-1][posx+3] > cb: 
	          if i[posy+-2][posx+2] > cb: 
	           if i[posy+-3][posx+1] > cb: 
	            if i[posy+-3][posx+0] > cb: 
	             return 1
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	       else:
	        if i[posy+0][posx+3] > cb: 
	         if i[posy+-1][posx+3] > cb: 
	          if i[posy+-2][posx+2] > cb: 
	           if i[posy+-3][posx+1] > cb: 
	            if i[posy+-3][posx+0] > cb: 
	             return 1
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	      else:
	       if i[posy+1][posx+3] > cb: 
	        if i[posy+0][posx+3] > cb: 
	         if i[posy+-1][posx+3] > cb: 
	          if i[posy+-2][posx+2] > cb: 
	           if i[posy+-3][posx+1] > cb: 
	            if i[posy+-3][posx+0] > cb: 
	             return 1
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	       else:
	        return 0
	     else:
	      return 0
	    else:
	     return 0
	   elif i[posy+-3][posx+-1] < c_b: 
	    if i[posy+-3][posx+1] < c_b:
	     if i[posy+-3][posx+0] < c_b:
	      if i[posy+-2][posx+-2] < c_b:
	       if i[posy+-2][posx+2] < c_b:
	        if i[posy+-1][posx+3] < c_b:
	         if i[posy+0][posx+3] < c_b:
	          if i[posy+1][posx+3] < c_b:
	           return 1
	          else:
	           if i[posy+-1][posx+-3] < c_b:
	            if i[posy+0][posx+-3] < c_b:
	             return 1
	            else:
	             return 0
	           else:
	            return 0
	         else:
	          if i[posy+-1][posx+-3] < c_b:
	           if i[posy+0][posx+-3] < c_b:
	            if i[posy+1][posx+-3] < c_b:
	             return 1
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	        else:
	         if i[posy+-1][posx+-3] < c_b:
	          if i[posy+0][posx+-3] < c_b:
	           if i[posy+1][posx+-3] < c_b:
	            if i[posy+2][posx+-2] < c_b:
	             return 1
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	       else:
	        if i[posy+-1][posx+-3] < c_b:
	         if i[posy+0][posx+-3] < c_b:
	          if i[posy+1][posx+-3] < c_b:
	           if i[posy+2][posx+-2] < c_b:
	            if i[posy+3][posx+-1] < c_b:
	             return 1
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	      else:
	       return 0
	     else:
	      return 0
	    else:
	     return 0
	   else:
	    return 0
	  else:
	   if i[posy+-3][posx+-1] > cb: 
	    if i[posy+-2][posx+-2] > cb: 
	     if i[posy+-1][posx+-3] > cb: 
	      if i[posy+0][posx+-3] > cb: 
	       if i[posy+1][posx+-3] > cb: 
	        if i[posy+2][posx+-2] > cb: 
	         if i[posy+3][posx+-1] > cb: 
	          return 1
	         else:
	          if i[posy+-2][posx+2] > cb: 
	           if i[posy+-3][posx+1] > cb: 
	            if i[posy+-3][posx+0] > cb: 
	             return 1
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	        else:
	         if i[posy+-1][posx+3] > cb: 
	          if i[posy+-2][posx+2] > cb: 
	           if i[posy+-3][posx+1] > cb: 
	            if i[posy+-3][posx+0] > cb: 
	             return 1
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	       else:
	        if i[posy+0][posx+3] > cb: 
	         if i[posy+-1][posx+3] > cb: 
	          if i[posy+-2][posx+2] > cb: 
	           if i[posy+-3][posx+1] > cb: 
	            if i[posy+-3][posx+0] > cb: 
	             return 1
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	      else:
	       if i[posy+1][posx+3] > cb: 
	        if i[posy+0][posx+3] > cb: 
	         if i[posy+-1][posx+3] > cb: 
	          if i[posy+-2][posx+2] > cb: 
	           if i[posy+-3][posx+1] > cb: 
	            if i[posy+-3][posx+0] > cb: 
	             return 1
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	       else:
	        return 0
	     else:
	      return 0
	    else:
	     return 0
	   elif i[posy+-3][posx+-1] < c_b: 
	    if i[posy+-3][posx+1] < c_b:
	     if i[posy+-3][posx+0] < c_b:
	      if i[posy+-2][posx+-2] < c_b:
	       if i[posy+-1][posx+-3] < c_b:
	        if i[posy+-2][posx+2] < c_b:
	         if i[posy+-1][posx+3] < c_b:
	          if i[posy+0][posx+3] < c_b:
	           if i[posy+1][posx+3] < c_b:
	            return 1
	           else:
	            if i[posy+0][posx+-3] < c_b:
	             return 1
	            else:
	             return 0
	          else:
	           if i[posy+0][posx+-3] < c_b:
	            if i[posy+1][posx+-3] < c_b:
	             return 1
	            else:
	             return 0
	           else:
	            return 0
	         else:
	          if i[posy+0][posx+-3] < c_b:
	           if i[posy+1][posx+-3] < c_b:
	            if i[posy+2][posx+-2] < c_b:
	             return 1
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	        else:
	         if i[posy+0][posx+-3] < c_b:
	          if i[posy+1][posx+-3] < c_b:
	           if i[posy+2][posx+-2] < c_b:
	            if i[posy+3][posx+-1] < c_b:
	             return 1
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	       else:
	        return 0
	      else:
	       return 0
	     else:
	      return 0
	    else:
	     return 0
	   else:
	    return 0
	 elif i[posy+3][posx+1] < c_b: 
	  if i[posy+-3][posx+0] > cb: 
	   if i[posy+-3][posx+-1] > cb: 
	    if i[posy+-2][posx+-2] > cb: 
	     if i[posy+-1][posx+-3] > cb: 
	      if i[posy+0][posx+-3] > cb: 
	       if i[posy+1][posx+-3] > cb: 
	        if i[posy+2][posx+-2] > cb: 
	         if i[posy+3][posx+-1] > cb: 
	          return 1
	         else:
	          if i[posy+-2][posx+2] > cb: 
	           if i[posy+-3][posx+1] > cb: 
	            return 1
	           else:
	            return 0
	          else:
	           return 0
	        else:
	         if i[posy+-1][posx+3] > cb: 
	          if i[posy+-2][posx+2] > cb: 
	           if i[posy+-3][posx+1] > cb: 
	            return 1
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	       else:
	        if i[posy+0][posx+3] > cb: 
	         if i[posy+-1][posx+3] > cb: 
	          if i[posy+-2][posx+2] > cb: 
	           if i[posy+-3][posx+1] > cb: 
	            return 1
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	      else:
	       if i[posy+1][posx+3] > cb: 
	        if i[posy+0][posx+3] > cb: 
	         if i[posy+-1][posx+3] > cb: 
	          if i[posy+-2][posx+2] > cb: 
	           if i[posy+-3][posx+1] > cb: 
	            return 1
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	       else:
	        return 0
	     else:
	      if i[posy+2][posx+2] > cb: 
	       if i[posy+1][posx+3] > cb: 
	        if i[posy+0][posx+3] > cb: 
	         if i[posy+-1][posx+3] > cb: 
	          if i[posy+-2][posx+2] > cb: 
	           if i[posy+-3][posx+1] > cb: 
	            return 1
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	       else:
	        return 0
	      else:
	       return 0
	    else:
	     return 0
	   else:
	    return 0
	  elif i[posy+-3][posx+0] < c_b: 
	   if i[posy+-3][posx+1] < c_b:
	    if i[posy+-3][posx+-1] < c_b:
	     if i[posy+-2][posx+2] < c_b:
	      if i[posy+-1][posx+3] < c_b:
	       if i[posy+0][posx+3] < c_b:
	        if i[posy+1][posx+3] < c_b:
	         if i[posy+2][posx+2] < c_b:
	          return 1
	         else:
	          if i[posy+-2][posx+-2] < c_b:
	           if i[posy+-1][posx+-3] < c_b:
	            return 1
	           else:
	            return 0
	          else:
	           return 0
	        else:
	         if i[posy+-2][posx+-2] < c_b:
	          if i[posy+-1][posx+-3] < c_b:
	           if i[posy+0][posx+-3] < c_b:
	            return 1
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	       else:
	        if i[posy+-2][posx+-2] < c_b:
	         if i[posy+-1][posx+-3] < c_b:
	          if i[posy+0][posx+-3] < c_b:
	           if i[posy+1][posx+-3] < c_b:
	            return 1
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	      else:
	       if i[posy+-2][posx+-2] < c_b:
	        if i[posy+-1][posx+-3] < c_b:
	         if i[posy+0][posx+-3] < c_b:
	          if i[posy+1][posx+-3] < c_b:
	           if i[posy+2][posx+-2] < c_b:
	            return 1
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	       else:
	        return 0
	     else:
	      if i[posy+-2][posx+-2] < c_b:
	       if i[posy+-1][posx+-3] < c_b:
	        if i[posy+0][posx+-3] < c_b:
	         if i[posy+1][posx+-3] < c_b:
	          if i[posy+2][posx+-2] < c_b:
	           if i[posy+3][posx+-1] < c_b:
	            return 1
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	       else:
	        return 0
	      else:
	       return 0
	    else:
	     return 0
	   else:
	    return 0
	  else:
	   return 0
	 else:
	  if i[posy+-3][posx+0] > cb: 
	   if i[posy+-3][posx+-1] > cb: 
	    if i[posy+-2][posx+-2] > cb: 
	     if i[posy+-1][posx+-3] > cb: 
	      if i[posy+0][posx+-3] > cb: 
	       if i[posy+1][posx+-3] > cb: 
	        if i[posy+2][posx+-2] > cb: 
	         if i[posy+3][posx+-1] > cb: 
	          return 1
	         else:
	          if i[posy+-2][posx+2] > cb: 
	           if i[posy+-3][posx+1] > cb: 
	            return 1
	           else:
	            return 0
	          else:
	           return 0
	        else:
	         if i[posy+-1][posx+3] > cb: 
	          if i[posy+-2][posx+2] > cb: 
	           if i[posy+-3][posx+1] > cb: 
	            return 1
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	       else:
	        if i[posy+0][posx+3] > cb: 
	         if i[posy+-1][posx+3] > cb: 
	          if i[posy+-2][posx+2] > cb: 
	           if i[posy+-3][posx+1] > cb: 
	            return 1
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	      else:
	       if i[posy+1][posx+3] > cb: 
	        if i[posy+0][posx+3] > cb: 
	         if i[posy+-1][posx+3] > cb: 
	          if i[posy+-2][posx+2] > cb: 
	           if i[posy+-3][posx+1] > cb: 
	            return 1
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	       else:
	        return 0
	     else:
	      if i[posy+2][posx+2] > cb: 
	       if i[posy+1][posx+3] > cb: 
	        if i[posy+0][posx+3] > cb: 
	         if i[posy+-1][posx+3] > cb: 
	          if i[posy+-2][posx+2] > cb: 
	           if i[posy+-3][posx+1] > cb: 
	            return 1
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	       else:
	        return 0
	      else:
	       return 0
	    else:
	     return 0
	   else:
	    return 0
	  elif i[posy+-3][posx+0] < c_b: 
	   if i[posy+-3][posx+1] < c_b:
	    if i[posy+-3][posx+-1] < c_b:
	     if i[posy+-2][posx+-2] < c_b:
	      if i[posy+-2][posx+2] < c_b:
	       if i[posy+-1][posx+3] < c_b:
	        if i[posy+0][posx+3] < c_b:
	         if i[posy+1][posx+3] < c_b:
	          if i[posy+2][posx+2] < c_b:
	           return 1
	          else:
	           if i[posy+-1][posx+-3] < c_b:
	            return 1
	           else:
	            return 0
	         else:
	          if i[posy+-1][posx+-3] < c_b:
	           if i[posy+0][posx+-3] < c_b:
	            return 1
	           else:
	            return 0
	          else:
	           return 0
	        else:
	         if i[posy+-1][posx+-3] < c_b:
	          if i[posy+0][posx+-3] < c_b:
	           if i[posy+1][posx+-3] < c_b:
	            return 1
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	       else:
	        if i[posy+-1][posx+-3] < c_b:
	         if i[posy+0][posx+-3] < c_b:
	          if i[posy+1][posx+-3] < c_b:
	           if i[posy+2][posx+-2] < c_b:
	            return 1
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	      else:
	       if i[posy+-1][posx+-3] < c_b:
	        if i[posy+0][posx+-3] < c_b:
	         if i[posy+1][posx+-3] < c_b:
	          if i[posy+2][posx+-2] < c_b:
	           if i[posy+3][posx+-1] < c_b:
	            return 1
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	       else:
	        return 0
	     else:
	      return 0
	    else:
	     return 0
	   else:
	    return 0
	  else:
	   return 0
	elif i[posy+3][posx+0] < c_b: 
	 if i[posy+3][posx+1] > cb: 
	  if i[posy+-3][posx+0] > cb: 
	   if i[posy+-3][posx+1] > cb: 
	    if i[posy+-3][posx+-1] > cb: 
	     if i[posy+-2][posx+2] > cb: 
	      if i[posy+-1][posx+3] > cb: 
	       if i[posy+0][posx+3] > cb: 
	        if i[posy+1][posx+3] > cb: 
	         if i[posy+2][posx+2] > cb: 
	          return 1
	         else:
	          if i[posy+-2][posx+-2] > cb: 
	           if i[posy+-1][posx+-3] > cb: 
	            return 1
	           else:
	            return 0
	          else:
	           return 0
	        else:
	         if i[posy+-2][posx+-2] > cb: 
	          if i[posy+-1][posx+-3] > cb: 
	           if i[posy+0][posx+-3] > cb: 
	            return 1
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	       else:
	        if i[posy+-2][posx+-2] > cb: 
	         if i[posy+-1][posx+-3] > cb: 
	          if i[posy+0][posx+-3] > cb: 
	           if i[posy+1][posx+-3] > cb: 
	            return 1
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	      else:
	       if i[posy+-2][posx+-2] > cb: 
	        if i[posy+-1][posx+-3] > cb: 
	         if i[posy+0][posx+-3] > cb: 
	          if i[posy+1][posx+-3] > cb: 
	           if i[posy+2][posx+-2] > cb: 
	            return 1
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	       else:
	        return 0
	     else:
	      if i[posy+-2][posx+-2] > cb: 
	       if i[posy+-1][posx+-3] > cb: 
	        if i[posy+0][posx+-3] > cb: 
	         if i[posy+1][posx+-3] > cb: 
	          if i[posy+2][posx+-2] > cb: 
	           if i[posy+3][posx+-1] > cb: 
	            return 1
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	       else:
	        return 0
	      else:
	       return 0
	    else:
	     return 0
	   else:
	    return 0
	  elif i[posy+-3][posx+0] < c_b: 
	   if i[posy+-3][posx+-1] < c_b:
	    if i[posy+-2][posx+-2] < c_b:
	     if i[posy+-1][posx+-3] < c_b:
	      if i[posy+0][posx+-3] < c_b:
	       if i[posy+1][posx+-3] < c_b:
	        if i[posy+2][posx+-2] < c_b:
	         if i[posy+3][posx+-1] < c_b:
	          return 1
	         else:
	          if i[posy+-2][posx+2] < c_b:
	           if i[posy+-3][posx+1] < c_b:
	            return 1
	           else:
	            return 0
	          else:
	           return 0
	        else:
	         if i[posy+-1][posx+3] < c_b:
	          if i[posy+-2][posx+2] < c_b:
	           if i[posy+-3][posx+1] < c_b:
	            return 1
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	       else:
	        if i[posy+0][posx+3] < c_b:
	         if i[posy+-1][posx+3] < c_b:
	          if i[posy+-2][posx+2] < c_b:
	           if i[posy+-3][posx+1] < c_b:
	            return 1
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	      else:
	       if i[posy+1][posx+3] < c_b:
	        if i[posy+0][posx+3] < c_b:
	         if i[posy+-1][posx+3] < c_b:
	          if i[posy+-2][posx+2] < c_b:
	           if i[posy+-3][posx+1] < c_b:
	            return 1
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	       else:
	        return 0
	     else:
	      if i[posy+2][posx+2] < c_b:
	       if i[posy+1][posx+3] < c_b:
	        if i[posy+0][posx+3] < c_b:
	         if i[posy+-1][posx+3] < c_b:
	          if i[posy+-2][posx+2] < c_b:
	           if i[posy+-3][posx+1] < c_b:
	            return 1
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	       else:
	        return 0
	      else:
	       return 0
	    else:
	     return 0
	   else:
	    return 0
	  else:
	   return 0
	 elif i[posy+3][posx+1] < c_b: 
	  if i[posy+2][posx+2] > cb: 
	   if i[posy+-3][posx+-1] > cb: 
	    if i[posy+-3][posx+1] > cb: 
	     if i[posy+-3][posx+0] > cb: 
	      if i[posy+-2][posx+-2] > cb: 
	       if i[posy+-2][posx+2] > cb: 
	        if i[posy+-1][posx+3] > cb: 
	         if i[posy+0][posx+3] > cb: 
	          if i[posy+1][posx+3] > cb: 
	           return 1
	          else:
	           if i[posy+-1][posx+-3] > cb: 
	            if i[posy+0][posx+-3] > cb: 
	             return 1
	            else:
	             return 0
	           else:
	            return 0
	         else:
	          if i[posy+-1][posx+-3] > cb: 
	           if i[posy+0][posx+-3] > cb: 
	            if i[posy+1][posx+-3] > cb: 
	             return 1
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	        else:
	         if i[posy+-1][posx+-3] > cb: 
	          if i[posy+0][posx+-3] > cb: 
	           if i[posy+1][posx+-3] > cb: 
	            if i[posy+2][posx+-2] > cb: 
	             return 1
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	       else:
	        if i[posy+-1][posx+-3] > cb: 
	         if i[posy+0][posx+-3] > cb: 
	          if i[posy+1][posx+-3] > cb: 
	           if i[posy+2][posx+-2] > cb: 
	            if i[posy+3][posx+-1] > cb: 
	             return 1
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	      else:
	       return 0
	     else:
	      return 0
	    else:
	     return 0
	   elif i[posy+-3][posx+-1] < c_b: 
	    if i[posy+-2][posx+-2] < c_b:
	     if i[posy+-1][posx+-3] < c_b:
	      if i[posy+0][posx+-3] < c_b:
	       if i[posy+1][posx+-3] < c_b:
	        if i[posy+2][posx+-2] < c_b:
	         if i[posy+3][posx+-1] < c_b:
	          return 1
	         else:
	          if i[posy+-2][posx+2] < c_b:
	           if i[posy+-3][posx+1] < c_b:
	            if i[posy+-3][posx+0] < c_b:
	             return 1
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	        else:
	         if i[posy+-1][posx+3] < c_b:
	          if i[posy+-2][posx+2] < c_b:
	           if i[posy+-3][posx+1] < c_b:
	            if i[posy+-3][posx+0] < c_b:
	             return 1
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	       else:
	        if i[posy+0][posx+3] < c_b:
	         if i[posy+-1][posx+3] < c_b:
	          if i[posy+-2][posx+2] < c_b:
	           if i[posy+-3][posx+1] < c_b:
	            if i[posy+-3][posx+0] < c_b:
	             return 1
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	      else:
	       if i[posy+1][posx+3] < c_b:
	        if i[posy+0][posx+3] < c_b:
	         if i[posy+-1][posx+3] < c_b:
	          if i[posy+-2][posx+2] < c_b:
	           if i[posy+-3][posx+1] < c_b:
	            if i[posy+-3][posx+0] < c_b:
	             return 1
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	       else:
	        return 0
	     else:
	      return 0
	    else:
	     return 0
	   else:
	    return 0
	  elif i[posy+2][posx+2] < c_b: 
	   if i[posy+1][posx+3] > cb: 
	    if i[posy+-2][posx+-2] > cb: 
	     if i[posy+-3][posx+1] > cb: 
	      if i[posy+-3][posx+0] > cb: 
	       if i[posy+-3][posx+-1] > cb: 
	        if i[posy+-1][posx+-3] > cb: 
	         if i[posy+-2][posx+2] > cb: 
	          if i[posy+-1][posx+3] > cb: 
	           if i[posy+0][posx+3] > cb: 
	            return 1
	           else:
	            if i[posy+0][posx+-3] > cb: 
	             if i[posy+1][posx+-3] > cb: 
	              return 1
	             else:
	              return 0
	            else:
	             return 0
	          else:
	           if i[posy+0][posx+-3] > cb: 
	            if i[posy+1][posx+-3] > cb: 
	             if i[posy+2][posx+-2] > cb: 
	              return 1
	             else:
	              return 0
	            else:
	             return 0
	           else:
	            return 0
	         else:
	          if i[posy+0][posx+-3] > cb: 
	           if i[posy+1][posx+-3] > cb: 
	            if i[posy+2][posx+-2] > cb: 
	             if i[posy+3][posx+-1] > cb: 
	              return 1
	             else:
	              return 0
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	        else:
	         return 0
	       else:
	        return 0
	      else:
	       return 0
	     else:
	      return 0
	    elif i[posy+-2][posx+-2] < c_b: 
	     if i[posy+-1][posx+-3] < c_b:
	      if i[posy+0][posx+-3] < c_b:
	       if i[posy+1][posx+-3] < c_b:
	        if i[posy+2][posx+-2] < c_b:
	         if i[posy+3][posx+-1] < c_b:
	          return 1
	         else:
	          if i[posy+-2][posx+2] < c_b:
	           if i[posy+-3][posx+1] < c_b:
	            if i[posy+-3][posx+0] < c_b:
	             if i[posy+-3][posx+-1] < c_b:
	              return 1
	             else:
	              return 0
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	        else:
	         if i[posy+-1][posx+3] < c_b:
	          if i[posy+-2][posx+2] < c_b:
	           if i[posy+-3][posx+1] < c_b:
	            if i[posy+-3][posx+0] < c_b:
	             if i[posy+-3][posx+-1] < c_b:
	              return 1
	             else:
	              return 0
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	       else:
	        if i[posy+0][posx+3] < c_b:
	         if i[posy+-1][posx+3] < c_b:
	          if i[posy+-2][posx+2] < c_b:
	           if i[posy+-3][posx+1] < c_b:
	            if i[posy+-3][posx+0] < c_b:
	             if i[posy+-3][posx+-1] < c_b:
	              return 1
	             else:
	              return 0
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	      else:
	       return 0
	     else:
	      return 0
	    else:
	     return 0
	   elif i[posy+1][posx+3] < c_b: 
	    if i[posy+0][posx+3] > cb: 
	     if i[posy+1][posx+-3] > cb: 
	      if i[posy+-3][posx+1] > cb: 
	       if i[posy+-3][posx+0] > cb: 
	        if i[posy+-3][posx+-1] > cb: 
	         if i[posy+-2][posx+-2] > cb: 
	          if i[posy+-1][posx+-3] > cb: 
	           if i[posy+0][posx+-3] > cb: 
	            if i[posy+-2][posx+2] > cb: 
	             if i[posy+-1][posx+3] > cb: 
	              return 1
	             else:
	              if i[posy+2][posx+-2] > cb: 
	               return 1
	              else:
	               return 0
	            else:
	             if i[posy+2][posx+-2] > cb: 
	              if i[posy+3][posx+-1] > cb: 
	               return 1
	              else:
	               return 0
	             else:
	              return 0
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	       else:
	        return 0
	      else:
	       return 0
	     elif i[posy+1][posx+-3] < c_b: 
	      if i[posy+-1][posx+-3] > cb: 
	       if i[posy+-1][posx+3] > cb: 
	        if i[posy+-2][posx+2] > cb: 
	         if i[posy+-3][posx+1] > cb: 
	          if i[posy+-3][posx+0] > cb: 
	           if i[posy+-3][posx+-1] > cb: 
	            if i[posy+-2][posx+-2] > cb: 
	             if i[posy+0][posx+-3] > cb: 
	              return 1
	             else:
	              return 0
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	       else:
	        return 0
	      elif i[posy+-1][posx+-3] < c_b: 
	       if i[posy+0][posx+-3] < c_b:
	        if i[posy+2][posx+-2] < c_b:
	         if i[posy+3][posx+-1] < c_b:
	          return 1
	         else:
	          if i[posy+-2][posx+2] < c_b:
	           if i[posy+-3][posx+1] < c_b:
	            if i[posy+-3][posx+0] < c_b:
	             if i[posy+-3][posx+-1] < c_b:
	              if i[posy+-2][posx+-2] < c_b:
	               return 1
	              else:
	               return 0
	             else:
	              return 0
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	        else:
	         if i[posy+-1][posx+3] < c_b:
	          if i[posy+-2][posx+2] < c_b:
	           if i[posy+-3][posx+1] < c_b:
	            if i[posy+-3][posx+0] < c_b:
	             if i[posy+-3][posx+-1] < c_b:
	              if i[posy+-2][posx+-2] < c_b:
	               return 1
	              else:
	               return 0
	             else:
	              return 0
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	       else:
	        return 0
	      else:
	       return 0
	     else:
	      if i[posy+-1][posx+3] > cb: 
	       if i[posy+-2][posx+2] > cb: 
	        if i[posy+-3][posx+1] > cb: 
	         if i[posy+-3][posx+0] > cb: 
	          if i[posy+-3][posx+-1] > cb: 
	           if i[posy+-2][posx+-2] > cb: 
	            if i[posy+-1][posx+-3] > cb: 
	             if i[posy+0][posx+-3] > cb: 
	              return 1
	             else:
	              return 0
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	       else:
	        return 0
	      else:
	       return 0
	    elif i[posy+0][posx+3] < c_b: 
	     if i[posy+-1][posx+3] > cb: 
	      if i[posy+2][posx+-2] > cb: 
	       if i[posy+-3][posx+1] > cb: 
	        if i[posy+-3][posx+0] > cb: 
	         if i[posy+-3][posx+-1] > cb: 
	          if i[posy+-2][posx+-2] > cb: 
	           if i[posy+-1][posx+-3] > cb: 
	            if i[posy+0][posx+-3] > cb: 
	             if i[posy+1][posx+-3] > cb: 
	              if i[posy+-2][posx+2] > cb: 
	               return 1
	              else:
	               if i[posy+3][posx+-1] > cb: 
	                return 1
	               else:
	                return 0
	             else:
	              return 0
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	       else:
	        return 0
	      elif i[posy+2][posx+-2] < c_b: 
	       if i[posy+0][posx+-3] > cb: 
	        if i[posy+-2][posx+2] > cb: 
	         if i[posy+-3][posx+1] > cb: 
	          if i[posy+-3][posx+0] > cb: 
	           if i[posy+-3][posx+-1] > cb: 
	            if i[posy+-2][posx+-2] > cb: 
	             if i[posy+-1][posx+-3] > cb: 
	              if i[posy+1][posx+-3] > cb: 
	               return 1
	              else:
	               return 0
	             else:
	              return 0
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	       elif i[posy+0][posx+-3] < c_b: 
	        if i[posy+1][posx+-3] < c_b:
	         if i[posy+3][posx+-1] < c_b:
	          return 1
	         else:
	          if i[posy+-2][posx+2] < c_b:
	           if i[posy+-3][posx+1] < c_b:
	            if i[posy+-3][posx+0] < c_b:
	             if i[posy+-3][posx+-1] < c_b:
	              if i[posy+-2][posx+-2] < c_b:
	               if i[posy+-1][posx+-3] < c_b:
	                return 1
	               else:
	                return 0
	              else:
	               return 0
	             else:
	              return 0
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	        else:
	         return 0
	       else:
	        return 0
	      else:
	       if i[posy+-2][posx+2] > cb: 
	        if i[posy+-3][posx+1] > cb: 
	         if i[posy+-3][posx+0] > cb: 
	          if i[posy+-3][posx+-1] > cb: 
	           if i[posy+-2][posx+-2] > cb: 
	            if i[posy+-1][posx+-3] > cb: 
	             if i[posy+0][posx+-3] > cb: 
	              if i[posy+1][posx+-3] > cb: 
	               return 1
	              else:
	               return 0
	             else:
	              return 0
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	       else:
	        return 0
	     elif i[posy+-1][posx+3] < c_b: 
	      if i[posy+-2][posx+2] > cb: 
	       if i[posy+3][posx+-1] < c_b:
	        if i[posy+1][posx+-3] > cb: 
	         if i[posy+-3][posx+1] > cb: 
	          if i[posy+-3][posx+0] > cb: 
	           if i[posy+-3][posx+-1] > cb: 
	            if i[posy+-2][posx+-2] > cb: 
	             if i[posy+-1][posx+-3] > cb: 
	              if i[posy+0][posx+-3] > cb: 
	               if i[posy+2][posx+-2] > cb: 
	                return 1
	               else:
	                return 0
	              else:
	               return 0
	             else:
	              return 0
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	        elif i[posy+1][posx+-3] < c_b: 
	         if i[posy+2][posx+-2] < c_b:
	          return 1
	         else:
	          return 0
	        else:
	         return 0
	       else:
	        if i[posy+-3][posx+1] > cb: 
	         if i[posy+-3][posx+0] > cb: 
	          if i[posy+-3][posx+-1] > cb: 
	           if i[posy+-2][posx+-2] > cb: 
	            if i[posy+-1][posx+-3] > cb: 
	             if i[posy+0][posx+-3] > cb: 
	              if i[posy+1][posx+-3] > cb: 
	               if i[posy+2][posx+-2] > cb: 
	                return 1
	               else:
	                return 0
	              else:
	               return 0
	             else:
	              return 0
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	      elif i[posy+-2][posx+2] < c_b: 
	       if i[posy+-3][posx+1] > cb: 
	        if i[posy+2][posx+-2] > cb: 
	         if i[posy+-3][posx+0] > cb: 
	          if i[posy+-3][posx+-1] > cb: 
	           if i[posy+-2][posx+-2] > cb: 
	            if i[posy+-1][posx+-3] > cb: 
	             if i[posy+0][posx+-3] > cb: 
	              if i[posy+1][posx+-3] > cb: 
	               if i[posy+3][posx+-1] > cb: 
	                return 1
	               else:
	                return 0
	              else:
	               return 0
	             else:
	              return 0
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	        elif i[posy+2][posx+-2] < c_b: 
	         if i[posy+3][posx+-1] < c_b:
	          return 1
	         else:
	          return 0
	        else:
	         return 0
	       elif i[posy+-3][posx+1] < c_b: 
	        if i[posy+-3][posx+0] < c_b:
	         return 1
	        else:
	         if i[posy+3][posx+-1] < c_b:
	          return 1
	         else:
	          return 0
	       else:
	        if i[posy+2][posx+-2] < c_b:
	         if i[posy+3][posx+-1] < c_b:
	          return 1
	         else:
	          return 0
	        else:
	         return 0
	      else:
	       if i[posy+1][posx+-3] > cb: 
	        if i[posy+-3][posx+1] > cb: 
	         if i[posy+-3][posx+0] > cb: 
	          if i[posy+-3][posx+-1] > cb: 
	           if i[posy+-2][posx+-2] > cb: 
	            if i[posy+-1][posx+-3] > cb: 
	             if i[posy+0][posx+-3] > cb: 
	              if i[posy+2][posx+-2] > cb: 
	               if i[posy+3][posx+-1] > cb: 
	                return 1
	               else:
	                return 0
	              else:
	               return 0
	             else:
	              return 0
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	       elif i[posy+1][posx+-3] < c_b: 
	        if i[posy+2][posx+-2] < c_b:
	         if i[posy+3][posx+-1] < c_b:
	          return 1
	         else:
	          return 0
	        else:
	         return 0
	       else:
	        return 0
	     else:
	      if i[posy+0][posx+-3] > cb: 
	       if i[posy+-3][posx+1] > cb: 
	        if i[posy+-3][posx+0] > cb: 
	         if i[posy+-3][posx+-1] > cb: 
	          if i[posy+-2][posx+-2] > cb: 
	           if i[posy+-1][posx+-3] > cb: 
	            if i[posy+1][posx+-3] > cb: 
	             if i[posy+2][posx+-2] > cb: 
	              if i[posy+-2][posx+2] > cb: 
	               return 1
	              else:
	               if i[posy+3][posx+-1] > cb: 
	                return 1
	               else:
	                return 0
	             else:
	              return 0
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	       else:
	        return 0
	      elif i[posy+0][posx+-3] < c_b: 
	       if i[posy+1][posx+-3] < c_b:
	        if i[posy+2][posx+-2] < c_b:
	         if i[posy+3][posx+-1] < c_b:
	          return 1
	         else:
	          if i[posy+-2][posx+2] < c_b:
	           if i[posy+-3][posx+1] < c_b:
	            if i[posy+-3][posx+0] < c_b:
	             if i[posy+-3][posx+-1] < c_b:
	              if i[posy+-2][posx+-2] < c_b:
	               if i[posy+-1][posx+-3] < c_b:
	                return 1
	               else:
	                return 0
	              else:
	               return 0
	             else:
	              return 0
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	        else:
	         return 0
	       else:
	        return 0
	      else:
	       return 0
	    else:
	     if i[posy+-1][posx+-3] > cb: 
	      if i[posy+-3][posx+1] > cb: 
	       if i[posy+-3][posx+0] > cb: 
	        if i[posy+-3][posx+-1] > cb: 
	         if i[posy+-2][posx+-2] > cb: 
	          if i[posy+0][posx+-3] > cb: 
	           if i[posy+1][posx+-3] > cb: 
	            if i[posy+-2][posx+2] > cb: 
	             if i[posy+-1][posx+3] > cb: 
	              return 1
	             else:
	              if i[posy+2][posx+-2] > cb: 
	               return 1
	              else:
	               return 0
	            else:
	             if i[posy+2][posx+-2] > cb: 
	              if i[posy+3][posx+-1] > cb: 
	               return 1
	              else:
	               return 0
	             else:
	              return 0
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	       else:
	        return 0
	      else:
	       return 0
	     elif i[posy+-1][posx+-3] < c_b: 
	      if i[posy+0][posx+-3] < c_b:
	       if i[posy+1][posx+-3] < c_b:
	        if i[posy+2][posx+-2] < c_b:
	         if i[posy+3][posx+-1] < c_b:
	          return 1
	         else:
	          if i[posy+-2][posx+2] < c_b:
	           if i[posy+-3][posx+1] < c_b:
	            if i[posy+-3][posx+0] < c_b:
	             if i[posy+-3][posx+-1] < c_b:
	              if i[posy+-2][posx+-2] < c_b:
	               return 1
	              else:
	               return 0
	             else:
	              return 0
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	        else:
	         if i[posy+-1][posx+3] < c_b:
	          if i[posy+-2][posx+2] < c_b:
	           if i[posy+-3][posx+1] < c_b:
	            if i[posy+-3][posx+0] < c_b:
	             if i[posy+-3][posx+-1] < c_b:
	              if i[posy+-2][posx+-2] < c_b:
	               return 1
	              else:
	               return 0
	             else:
	              return 0
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	       else:
	        return 0
	      else:
	       return 0
	     else:
	      return 0
	   else:
	    if i[posy+-2][posx+-2] > cb: 
	     if i[posy+-3][posx+1] > cb: 
	      if i[posy+-3][posx+0] > cb: 
	       if i[posy+-3][posx+-1] > cb: 
	        if i[posy+-1][posx+-3] > cb: 
	         if i[posy+0][posx+-3] > cb: 
	          if i[posy+-2][posx+2] > cb: 
	           if i[posy+-1][posx+3] > cb: 
	            if i[posy+0][posx+3] > cb: 
	             return 1
	            else:
	             if i[posy+1][posx+-3] > cb: 
	              return 1
	             else:
	              return 0
	           else:
	            if i[posy+1][posx+-3] > cb: 
	             if i[posy+2][posx+-2] > cb: 
	              return 1
	             else:
	              return 0
	            else:
	             return 0
	          else:
	           if i[posy+1][posx+-3] > cb: 
	            if i[posy+2][posx+-2] > cb: 
	             if i[posy+3][posx+-1] > cb: 
	              return 1
	             else:
	              return 0
	            else:
	             return 0
	           else:
	            return 0
	         else:
	          return 0
	        else:
	         return 0
	       else:
	        return 0
	      else:
	       return 0
	     else:
	      return 0
	    elif i[posy+-2][posx+-2] < c_b: 
	     if i[posy+-1][posx+-3] < c_b:
	      if i[posy+0][posx+-3] < c_b:
	       if i[posy+1][posx+-3] < c_b:
	        if i[posy+2][posx+-2] < c_b:
	         if i[posy+3][posx+-1] < c_b:
	          return 1
	         else:
	          if i[posy+-2][posx+2] < c_b:
	           if i[posy+-3][posx+1] < c_b:
	            if i[posy+-3][posx+0] < c_b:
	             if i[posy+-3][posx+-1] < c_b:
	              return 1
	             else:
	              return 0
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	        else:
	         if i[posy+-1][posx+3] < c_b:
	          if i[posy+-2][posx+2] < c_b:
	           if i[posy+-3][posx+1] < c_b:
	            if i[posy+-3][posx+0] < c_b:
	             if i[posy+-3][posx+-1] < c_b:
	              return 1
	             else:
	              return 0
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	       else:
	        if i[posy+0][posx+3] < c_b:
	         if i[posy+-1][posx+3] < c_b:
	          if i[posy+-2][posx+2] < c_b:
	           if i[posy+-3][posx+1] < c_b:
	            if i[posy+-3][posx+0] < c_b:
	             if i[posy+-3][posx+-1] < c_b:
	              return 1
	             else:
	              return 0
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	      else:
	       return 0
	     else:
	      return 0
	    else:
	     return 0
	  else:
	   if i[posy+-3][posx+-1] > cb: 
	    if i[posy+-3][posx+1] > cb: 
	     if i[posy+-3][posx+0] > cb: 
	      if i[posy+-2][posx+-2] > cb: 
	       if i[posy+-1][posx+-3] > cb: 
	        if i[posy+-2][posx+2] > cb: 
	         if i[posy+-1][posx+3] > cb: 
	          if i[posy+0][posx+3] > cb: 
	           if i[posy+1][posx+3] > cb: 
	            return 1
	           else:
	            if i[posy+0][posx+-3] > cb: 
	             return 1
	            else:
	             return 0
	          else:
	           if i[posy+0][posx+-3] > cb: 
	            if i[posy+1][posx+-3] > cb: 
	             return 1
	            else:
	             return 0
	           else:
	            return 0
	         else:
	          if i[posy+0][posx+-3] > cb: 
	           if i[posy+1][posx+-3] > cb: 
	            if i[posy+2][posx+-2] > cb: 
	             return 1
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	        else:
	         if i[posy+0][posx+-3] > cb: 
	          if i[posy+1][posx+-3] > cb: 
	           if i[posy+2][posx+-2] > cb: 
	            if i[posy+3][posx+-1] > cb: 
	             return 1
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	       else:
	        return 0
	      else:
	       return 0
	     else:
	      return 0
	    else:
	     return 0
	   elif i[posy+-3][posx+-1] < c_b: 
	    if i[posy+-2][posx+-2] < c_b:
	     if i[posy+-1][posx+-3] < c_b:
	      if i[posy+0][posx+-3] < c_b:
	       if i[posy+1][posx+-3] < c_b:
	        if i[posy+2][posx+-2] < c_b:
	         if i[posy+3][posx+-1] < c_b:
	          return 1
	         else:
	          if i[posy+-2][posx+2] < c_b:
	           if i[posy+-3][posx+1] < c_b:
	            if i[posy+-3][posx+0] < c_b:
	             return 1
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	        else:
	         if i[posy+-1][posx+3] < c_b:
	          if i[posy+-2][posx+2] < c_b:
	           if i[posy+-3][posx+1] < c_b:
	            if i[posy+-3][posx+0] < c_b:
	             return 1
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	       else:
	        if i[posy+0][posx+3] < c_b:
	         if i[posy+-1][posx+3] < c_b:
	          if i[posy+-2][posx+2] < c_b:
	           if i[posy+-3][posx+1] < c_b:
	            if i[posy+-3][posx+0] < c_b:
	             return 1
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	      else:
	       if i[posy+1][posx+3] < c_b:
	        if i[posy+0][posx+3] < c_b:
	         if i[posy+-1][posx+3] < c_b:
	          if i[posy+-2][posx+2] < c_b:
	           if i[posy+-3][posx+1] < c_b:
	            if i[posy+-3][posx+0] < c_b:
	             return 1
	            else:
	             return 0
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	       else:
	        return 0
	     else:
	      return 0
	    else:
	     return 0
	   else:
	    return 0
	 else:
	  if i[posy+-3][posx+0] > cb: 
	   if i[posy+-3][posx+1] > cb: 
	    if i[posy+-3][posx+-1] > cb: 
	     if i[posy+-2][posx+-2] > cb: 
	      if i[posy+-2][posx+2] > cb: 
	       if i[posy+-1][posx+3] > cb: 
	        if i[posy+0][posx+3] > cb: 
	         if i[posy+1][posx+3] > cb: 
	          if i[posy+2][posx+2] > cb: 
	           return 1
	          else:
	           if i[posy+-1][posx+-3] > cb: 
	            return 1
	           else:
	            return 0
	         else:
	          if i[posy+-1][posx+-3] > cb: 
	           if i[posy+0][posx+-3] > cb: 
	            return 1
	           else:
	            return 0
	          else:
	           return 0
	        else:
	         if i[posy+-1][posx+-3] > cb: 
	          if i[posy+0][posx+-3] > cb: 
	           if i[posy+1][posx+-3] > cb: 
	            return 1
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	       else:
	        if i[posy+-1][posx+-3] > cb: 
	         if i[posy+0][posx+-3] > cb: 
	          if i[posy+1][posx+-3] > cb: 
	           if i[posy+2][posx+-2] > cb: 
	            return 1
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	      else:
	       if i[posy+-1][posx+-3] > cb: 
	        if i[posy+0][posx+-3] > cb: 
	         if i[posy+1][posx+-3] > cb: 
	          if i[posy+2][posx+-2] > cb: 
	           if i[posy+3][posx+-1] > cb: 
	            return 1
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	       else:
	        return 0
	     else:
	      return 0
	    else:
	     return 0
	   else:
	    return 0
	  elif i[posy+-3][posx+0] < c_b: 
	   if i[posy+-3][posx+-1] < c_b:
	    if i[posy+-2][posx+-2] < c_b:
	     if i[posy+-1][posx+-3] < c_b:
	      if i[posy+0][posx+-3] < c_b:
	       if i[posy+1][posx+-3] < c_b:
	        if i[posy+2][posx+-2] < c_b:
	         if i[posy+3][posx+-1] < c_b:
	          return 1
	         else:
	          if i[posy+-2][posx+2] < c_b:
	           if i[posy+-3][posx+1] < c_b:
	            return 1
	           else:
	            return 0
	          else:
	           return 0
	        else:
	         if i[posy+-1][posx+3] < c_b:
	          if i[posy+-2][posx+2] < c_b:
	           if i[posy+-3][posx+1] < c_b:
	            return 1
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	       else:
	        if i[posy+0][posx+3] < c_b:
	         if i[posy+-1][posx+3] < c_b:
	          if i[posy+-2][posx+2] < c_b:
	           if i[posy+-3][posx+1] < c_b:
	            return 1
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	      else:
	       if i[posy+1][posx+3] < c_b:
	        if i[posy+0][posx+3] < c_b:
	         if i[posy+-1][posx+3] < c_b:
	          if i[posy+-2][posx+2] < c_b:
	           if i[posy+-3][posx+1] < c_b:
	            return 1
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	       else:
	        return 0
	     else:
	      if i[posy+2][posx+2] < c_b:
	       if i[posy+1][posx+3] < c_b:
	        if i[posy+0][posx+3] < c_b:
	         if i[posy+-1][posx+3] < c_b:
	          if i[posy+-2][posx+2] < c_b:
	           if i[posy+-3][posx+1] < c_b:
	            return 1
	           else:
	            return 0
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	       else:
	        return 0
	      else:
	       return 0
	    else:
	     return 0
	   else:
	    return 0
	  else:
	   return 0
	else:
	 if i[posy+-3][posx+1] > cb: 
	  if i[posy+-3][posx+0] > cb: 
	   if i[posy+-3][posx+-1] > cb: 
	    if i[posy+-2][posx+2] > cb: 
	     if i[posy+-1][posx+3] > cb: 
	      if i[posy+0][posx+3] > cb: 
	       if i[posy+1][posx+3] > cb: 
	        if i[posy+2][posx+2] > cb: 
	         if i[posy+3][posx+1] > cb: 
	          return 1
	         else:
	          if i[posy+-2][posx+-2] > cb: 
	           return 1
	          else:
	           return 0
	        else:
	         if i[posy+-2][posx+-2] > cb: 
	          if i[posy+-1][posx+-3] > cb: 
	           return 1
	          else:
	           return 0
	         else:
	          return 0
	       else:
	        if i[posy+-2][posx+-2] > cb: 
	         if i[posy+-1][posx+-3] > cb: 
	          if i[posy+0][posx+-3] > cb: 
	           return 1
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	      else:
	       if i[posy+-2][posx+-2] > cb: 
	        if i[posy+-1][posx+-3] > cb: 
	         if i[posy+0][posx+-3] > cb: 
	          if i[posy+1][posx+-3] > cb: 
	           return 1
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	       else:
	        return 0
	     else:
	      if i[posy+-2][posx+-2] > cb: 
	       if i[posy+-1][posx+-3] > cb: 
	        if i[posy+0][posx+-3] > cb: 
	         if i[posy+1][posx+-3] > cb: 
	          if i[posy+2][posx+-2] > cb: 
	           return 1
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	       else:
	        return 0
	      else:
	       return 0
	    else:
	     if i[posy+-2][posx+-2] > cb: 
	      if i[posy+-1][posx+-3] > cb: 
	       if i[posy+0][posx+-3] > cb: 
	        if i[posy+1][posx+-3] > cb: 
	         if i[posy+2][posx+-2] > cb: 
	          if i[posy+3][posx+-1] > cb: 
	           return 1
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	       else:
	        return 0
	      else:
	       return 0
	     else:
	      return 0
	   else:
	    return 0
	  else:
	   return 0
	 elif i[posy+-3][posx+1] < c_b: 
	  if i[posy+-3][posx+0] < c_b:
	   if i[posy+-3][posx+-1] < c_b:
	    if i[posy+-2][posx+2] < c_b:
	     if i[posy+-1][posx+3] < c_b:
	      if i[posy+0][posx+3] < c_b:
	       if i[posy+1][posx+3] < c_b:
	        if i[posy+2][posx+2] < c_b:
	         if i[posy+3][posx+1] < c_b:
	          return 1
	         else:
	          if i[posy+-2][posx+-2] < c_b:
	           return 1
	          else:
	           return 0
	        else:
	         if i[posy+-2][posx+-2] < c_b:
	          if i[posy+-1][posx+-3] < c_b:
	           return 1
	          else:
	           return 0
	         else:
	          return 0
	       else:
	        if i[posy+-2][posx+-2] < c_b:
	         if i[posy+-1][posx+-3] < c_b:
	          if i[posy+0][posx+-3] < c_b:
	           return 1
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	      else:
	       if i[posy+-2][posx+-2] < c_b:
	        if i[posy+-1][posx+-3] < c_b:
	         if i[posy+0][posx+-3] < c_b:
	          if i[posy+1][posx+-3] < c_b:
	           return 1
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	       else:
	        return 0
	     else:
	      if i[posy+-2][posx+-2] < c_b:
	       if i[posy+-1][posx+-3] < c_b:
	        if i[posy+0][posx+-3] < c_b:
	         if i[posy+1][posx+-3] < c_b:
	          if i[posy+2][posx+-2] < c_b:
	           return 1
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	       else:
	        return 0
	      else:
	       return 0
	    else:
	     if i[posy+-2][posx+-2] < c_b:
	      if i[posy+-1][posx+-3] < c_b:
	       if i[posy+0][posx+-3] < c_b:
	        if i[posy+1][posx+-3] < c_b:
	         if i[posy+2][posx+-2] < c_b:
	          if i[posy+3][posx+-1] < c_b:
	           return 1
	          else:
	           return 0
	         else:
	          return 0
	        else:
	         return 0
	       else:
	        return 0
	      else:
	       return 0
	     else:
	      return 0
	   else:
	    return 0
	  else:
	   return 0
	 else:
	  return 0

def corner_score(i, posx, posy):
    
	bmin = 0
	bmax = 255
	b = (bmax + bmin)/2
    
	#Compute the score using binary search
	while True:
    
		if is_a_corner(i, posx, posy, b):
			bmin = b
		else:
			bmax = b
        
		if bmin == bmax - 1 or bmin == bmax:
			return bmin

		b = (bmin + bmax) / 2

