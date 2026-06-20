const sidebar=document.getElementById('sidebar');
const toggle=document.querySelector('.menu-toggle');
const closeSidebar=()=>{document.body.classList.remove('sidebar-open');toggle?.setAttribute('aria-expanded','false')};
const openSidebar=()=>{document.body.classList.add('sidebar-open');toggle?.setAttribute('aria-expanded','true');sidebar?.querySelector('.nav-link')?.focus()};
toggle?.addEventListener('click',()=>document.body.classList.contains('sidebar-open')?closeSidebar():openSidebar());
document.querySelectorAll('[data-sidebar-close]').forEach(el=>el.addEventListener('click',closeSidebar));
document.addEventListener('keydown',event=>{if(event.key==='Escape')closeSidebar()});
document.querySelectorAll('.side-nav .nav-link').forEach(link=>{if(link.pathname===location.pathname){link.classList.add('active');link.setAttribute('aria-current','page')}link.addEventListener('click',closeSidebar)});
