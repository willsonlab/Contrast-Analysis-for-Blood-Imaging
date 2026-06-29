/*
 * Macro to process multiple images in a folder and save line profile
 */

#@ File (label = "Input directory", style = "directory") input
#@ File (label = "Output directory", style = "directory") output
#@ String (label = "File suffix", value = ".tif") suffix

//makeLine(0, 210, 1850, 210);

	
processFolder(input);

// Recursively process files with the specified suffix
function processFolder(input) {
	list = getFileList(input);
	list = Array.sort(list);
	print((File.isDirectory(input)));
	print(input + File.separator + list[1]);
	//print(list.length);
	for (i = 0; i < list.length; i++) {

	/*	if (File.isDirectory(input + File.separator + list[i])) {
			print("if" );	
			processFolder(input + File.separator + list[i]);
			
		} else if (endsWith(list[i], suffix)) {
			print("else" );	
			processFile(input, output, list[i], i);

		}
		//print(list[i] );	
		
	*/	
		processFile(input, output, list[i], i);
		
	}
}

function processFile(input, output, file, count) {

	print(file);
	open(input + File.separator + file);
	print(file);
	run("Set Scale...", "distance=1842.0003 known=260 unit=mm");
//	setScale(0.1412, "mm", 0, 0);

	makeLine(0, 225, 1850, 210);
	setLineWidth(10);              // make the line thicker

	wait(1000);  // Pause for 2 seconds

	profile1 = getProfile();
	wait(1000);  // Pause for 2 seconds

	for (i = 0; i < profile1.length; i++) {
		setResult("Value", i, profile1[i]);
	}
	updateResults();

	// Clean file name for saving (remove extension, replace bad chars)
	cleanName = replace(file, ".tif", "");
	cleanName = replace(cleanName, ".", "_");

	// Save results
	saveAs("Measurements", output + File.separator + "data_" + cleanName + ".txt");
	close("*");

	print("Processing: " + input + File.separator + file);
	print("Saved to: " + output + File.separator + "data_" + cleanName + ".txt");
	
	//resetResults(); // Clear previous results
}
