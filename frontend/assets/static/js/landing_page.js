(function($) {          
	$(document).ready(function(){                    
		$(window).scroll(function(){                          
			if ($(this).scrollTop() > 400) {
				$('.fixed.menu').fadeIn(500);
			} else {
				$('.fixed.menu').fadeOut(500);
			}
		});
 	});
})(jQuery);